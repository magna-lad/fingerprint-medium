from graph_minutiae import GraphMinutiae
from model import *
from load_save import *
def run_complete_pipeline(analyzer, device='cpu'):
    """
    Run the complete fingerprint GNN pipeline.
    
    Args:
        analyzer: MinutiaeROCAnalyzer instance with feature vectors computed
        device: 'cuda' or 'cpu'
    """
    print("="*60)
    print("FINGERPRINT GNN VERIFICATION PIPELINE")
    print("="*60)
    
    analyzer.k_nearest_negihbors(k=4)
    # Step 1: Build graphs
    print("\n[Step 1/8] Building graphs...")
    fingerprint_graphs = analyzer.graph_maker()
    
    # Step 2: Create pairs
    print("\n[Step 2/8] Creating graph pairs...")
    all_pairs = analyzer.create_graph_pairs(num_impostor_per_genuine=1)
    print(f"\nTotal fingerprints (graphs) extracted: {len(fingerprint_graphs)} (should be {len(users)*8*5})")
    if len(fingerprint_graphs) != len(users)*8*5:
        print("⚠ Mismatch between expected and actual number of graphs! Check data pipeline.")

    
    # Step 3: Split data
    print("\n[Step 3/8] Splitting data...")
    train_pairs, val_pairs, test_pairs = analyzer.split_pairs_train_val_test(
        all_pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    train_users, val_users, test_users = analyzer.get_user_splits()
    
    # Create the actual pair sets for val and test
    #train_pairs, val_pairs, test_pairs = analyzer.split_pairs_by_user(all_pairs, train_users, val_users, test_users)


    #train_user_ids = set(info['user_id'] for info in analyzer.fingerprint_graphs 
                         #if any(pair[0].graph_id == info['graph_id'] for pair in train_pairs))

    
    # Step 4: Create training triplets using ONLY the training users
    print("\n[Step 4/8] Creating triplets for training...")
    #train_triplets = analyzer.create_triplets(train_user_ids)
    train_graphs = [info for info in analyzer.fingerprint_graphs if info['user_id'] in train_users]
    # Create a similar list of graphs for the validation users
    val_graphs = [info for info in analyzer.fingerprint_graphs if info['user_id'] in val_users] # <-- ADD THIS LIN


    # Step 4: Initialize model
    print("\n[Step 4/8] Initializing model...")
    model = initialize_model(device="cpu")
    
    
    # Step 5: Train model
    print("\n[Step 5/8] Training model...")
    trained_model, history = train_model(
        model, train_graphs, val_graphs,
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        device="cpu"
    )
    
    # Step 6: Plot training history
    print("\n[Step 6/8] Plotting training history...")
    plot_training_history(history)
    
    # Step 7: Evaluate on test set
    print("\n[Step 7/8] Evaluating on test set...")
    results = evaluate_model(trained_model, test_pairs, device="cpu")
    
    # Step 8: Save final model
    print("\n[Step 8/8] Saving final model...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'results': results,
        'config': {
            'node_features': 4,
            'hidden_dim': 128,
            'output_dim': 256,
            'num_conv_layers': 4
        }
    }, 'final_fingerprint_gnn.pth')
    
    print("\n✓ Pipeline complete!")
    print(f"✓ Operational threshold: {results['eer_threshold']:.4f}")
    print(f"✓ Expected accuracy: {results['accuracy']*100:.2f}%")
    print(f"✓ Model saved to: final_fingerprint_gnn.pth")
    
    return trained_model, results


def plot_training_history(history):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['val_avg_genuine_dist'], label='Train Avg Distance', linewidth=2)
    ax2.plot(history['val_avg_impostor_dist'], label='Val Avg Distance', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Average Distance', fontsize=12)
    ax2.set_title('Average Embedding Distance', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()



users=load_users_dictionary('processed_minutiae_data.pkl',True)
print("---- Data Coverage Summary ----")
print(f"Total users loaded: {len(users)}")

missing_finger_user = False
missing_impression_user = False
for uid, udata in list(users.items())[:5]:  # Check first 5 users; increase if needed
    print(f"\nUser: {uid}")
    total_impressions = 0
    for hand in ["L", "R"]:
        fingers = udata['fingers'][hand]
        print(f"  {hand} hand -- Fingers: {len(fingers)} (should be <= 4 if split evenly)")
        for idx, finger in enumerate(fingers):
            n_impr = len(finger)
            print(f"    Finger {idx}: {n_impr} impressions")
            total_impressions += n_impr
            if n_impr != 5:
                missing_impression_user = True
    print(f"  Total impressions for {uid}: {total_impressions} (should be 40, got {total_impressions})")
    if total_impressions != 40:
        missing_finger_user = True

print("\n==== Summary for ALL users ====")
all_fingers = 0
all_impressions = 0
for uid, udata in users.items():
    for hand in ["L", "R"]:
        fingers = udata['fingers'][hand]
        all_fingers += len(fingers)
        for finger in fingers:
            all_impressions += len(finger)
print(f"Total fingers (should be 8 × users): {all_fingers}")
print(f"Total impressions (should be 40 × users): {all_impressions}")

print("\n  Should be 8 fingers × 5 impressions per user = 40 fingerprints per user")
print(f"  Expected total fingerprints (graphs) for {len(users)} users: {len(users)*8*5}")

if missing_finger_user or missing_impression_user:
    print("⚠ WARNING: Some users/fingers have missing or extra impressions!")
else:
    print("✓ All users and impressions accounted for.")

analyse= GraphMinutiae(users)
run_complete_pipeline(analyse)