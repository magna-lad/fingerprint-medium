import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt

class EdgeUpdateConv(MessagePassing):
    """Custom GNN layer with edge feature incorporation."""
    
    def __init__(self, in_channels, out_channels, edge_dim=2):
        super().__init__(aggr='mean')
        self.linear_node = Linear(out_channels, out_channels)
        self.linear_edge = Linear(in_channels + edge_dim, out_channels)
        self.bn = BatchNorm1d(out_channels)
        self.residual = Linear(in_channels, out_channels) if in_channels != out_channels else None


    

    def forward(self, x, edge_index, edge_attr):
        x_residual = x
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn(out)
        
        if self.residual is not None:
            x_residual = self.residual(x_residual)

        
        return F.relu(out + x_residual)

    def message(self, x_j, edge_attr):
        combined = torch.cat([x_j, edge_attr], dim=1)
        print(f"x_j: {x_j.shape}, edge_attr: {edge_attr.shape}, combined: {combined.shape}")
        return self.linear_edge(combined)

    def update(self, aggr_out):
        return self.linear_node(aggr_out)


class FingerprintDescriptorGNN(torch.nn.Module):
    """GNN model for fingerprint embedding."""
    
    def __init__(self, node_features=4, hidden_dim=128, output_dim=256, 
                 num_conv_layers=4, dropout=0.3):
        super().__init__()
        
        self.num_conv_layers = num_conv_layers
        
        # Graph convolution layers
        self.convs = torch.nn.ModuleList([
            EdgeUpdateConv(4, 128),  # first layer
            EdgeUpdateConv(128, 128),
            EdgeUpdateConv(128, 128),
            EdgeUpdateConv(128, 128)
        ])
        self.convs.append(EdgeUpdateConv(node_features, hidden_dim))
        for _ in range(num_conv_layers - 1):
            self.convs.append(EdgeUpdateConv(hidden_dim, hidden_dim))
        
        self.dropout_layers = torch.nn.ModuleList([
            Dropout(dropout) for _ in range(num_conv_layers)
        ])
        
        # Global pooling
        self.pool_mean = global_mean_pool
        self.pool_max = global_max_pool
        
        # MLP head
        self.fc1 = Linear(2 * hidden_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.dropout_layers[i](x)
        
        # Global pooling
        x_mean = self.pool_mean(x, batch)
        x_max = self.pool_max(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # MLP head
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        
        return x

class FingerprintContrastiveLoss(torch.nn.Module):
    """Contrastive loss for fingerprint pairs."""
    
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        
        # Genuine: minimize distance
        loss_genuine = label * distance.pow(2)
        
        # Impostor: maximize distance (penalty if < margin)
        loss_impostor = (1 - label) * F.relu(self.margin - distance).pow(2)
        
        loss = (loss_genuine + loss_impostor).mean()
        
        return loss, distance.mean().item()




def initialize_model(device='cuda'):
    """Initialize model and move to device."""
    model = FingerprintDescriptorGNN(
        node_features=4,
        hidden_dim=128,
        output_dim=256,
        num_conv_layers=4,
        dropout=0.3
    )
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✓ Model initialized with {num_params:,} trainable parameters")
    print(f"✓ Device: {device}")
    
    return model




def train_model(model, train_pairs, val_pairs, num_epochs=50, 
                batch_size=32, learning_rate=1e-3, device='cuda'):
    """
    Train the fingerprint GNN model.
    
    Args:
        model: FingerprintDescriptorGNN instance
        train_pairs: List of (graph1, graph2, label) for training
        val_pairs: List of (graph1, graph2, label) for validation
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        Trained model and training history
    """
    criterion = FingerprintContrastiveLoss(margin=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_distance': [],
        'val_distance': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    print(f"\nStarting training...")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    for epoch in range(num_epochs):
        # ===== TRAINING =====
        model.train()
        train_losses = []
        train_distances = []
        
        # Shuffle training data
        random.shuffle(train_pairs)
        
        # Process in batches
        for i in tqdm.tqdm((range(0, len(train_pairs), batch_size)), 
                     desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            batch = train_pairs[i:i+batch_size]
            
            batch_loss = 0
            batch_dist = 0
            
            for graph1, graph2, label in batch:
                # Move to device
                graph1 = graph1.to(device)
                graph2 = graph2.to(device)
                
                # Forward pass
                emb1 = model(graph1)
                emb2 = model(graph2)
                
                # Compute loss
                loss, avg_dist = criterion(
                    emb1, emb2, 
                    torch.tensor([label], dtype=torch.float, device=device)
                )
                
                batch_loss += loss
                batch_dist += avg_dist
            
            # Average over batch
            batch_loss /= len(batch)
            batch_dist /= len(batch)
            
            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(batch_loss.item())
            train_distances.append(batch_dist)
        
        avg_train_loss = np.mean(train_losses)
        avg_train_dist = np.mean(train_distances)
        
        # ===== VALIDATION =====
        model.eval()
        val_losses = []
        val_distances = []
        
        with torch.no_grad():
            for graph1, graph2, label in tqdm(val_pairs, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                graph1 = graph1.to(device)
                graph2 = graph2.to(device)
                
                emb1 = model(graph1)
                emb2 = model(graph2)
                
                loss, avg_dist = criterion(
                    emb1, emb2,
                    torch.tensor([label], dtype=torch.float, device=device)
                )
                
                val_losses.append(loss.item())
                val_distances.append(avg_dist)
        
        avg_val_loss = np.mean(val_losses)
        avg_val_dist = np.mean(val_distances)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_distance'].append(avg_train_dist)
        history['val_distance'].append(avg_val_dist)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Dist: {avg_train_dist:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Dist:   {avg_val_dist:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_fingerprint_gnn.pth')
            patience_counter = 0
            print(f"  ✓ Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    checkpoint = torch.load('best_fingerprint_gnn.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n✓ Training complete! Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, history



def evaluate_model(model, test_pairs, device='cuda'):
    """
    Evaluate model and compute ROC metrics.
    
    Args:
        model: Trained model
        test_pairs: List of test pairs
        device: Computation device
    
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import roc_curve, auc
    
    model.eval()
    
    distances = []
    labels = []
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for graph1, graph2, label in tqdm(test_pairs, desc="Computing embeddings"):
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            
            emb1 = model(graph1)
            emb2 = model(graph2)
            
            dist = F.pairwise_distance(emb1, emb2).item()
            
            distances.append(dist)
            labels.append(label)
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    # Compute ROC (use negative distance for ROC)
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    auc_score = auc(fpr, tpr)
    
    # Compute EER
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_index]
    eer_threshold = -thresholds[eer_index]
    
    # Accuracy at EER threshold
    predictions = (distances <= eer_threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    
    # Statistics
    genuine_distances = distances[labels == 1]
    impostor_distances = distances[labels == 0]
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"AUC Score:          {auc_score:.4f}")
    print(f"Equal Error Rate:   {eer:.4f} ({eer*100:.2f}%)")
    print(f"EER Threshold:      {eer_threshold:.4f}")
    print(f"Accuracy at EER:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nGenuine pairs:  Mean={genuine_distances.mean():.4f}, Std={genuine_distances.std():.4f}")
    print(f"Impostor pairs: Mean={impostor_distances.mean():.4f}, Std={impostor_distances.std():.4f}")
    print("="*60)
    
    # Plot ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.plot(fpr[eer_index], tpr[eer_index], 'go', markersize=12, 
             label=f'EER={eer:.4f}')
    plt.xlabel('False Accept Rate', fontsize=14)
    plt.ylabel('True Accept Rate', fontsize=14)
    plt.title('ROC Curve - Fingerprint Verification', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=300)
    plt.show()
    
    # Plot distance distributions
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_distances, bins=50, alpha=0.6, label='Genuine', color='green', density=True)
    plt.hist(impostor_distances, bins=50, alpha=0.6, label='Impostor', color='red', density=True)
    plt.axvline(eer_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'EER Threshold={eer_threshold:.3f}')
    plt.xlabel('Distance', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Distance Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('distance_distribution.png', dpi=300)
    plt.show()
    
    return {
        'auc': auc_score,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'accuracy': accuracy,
        'genuine_distances': genuine_distances,
        'impostor_distances': impostor_distances
    }
