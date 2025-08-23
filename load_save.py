import os
import pickle


def load_users_dictionary(filename,calc_roc=False):
    """Load the saved users dictionary"""
    
    cache_dir = "biometric_cache"
    filepath = os.path.join(cache_dir, filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    try:
        # Load with pickle
        with open(filepath, 'rb') as f:
            users = pickle.load(f)
        
        print(f"Users dictionary loaded successfully!")
        print(f"Location: {filepath}")
        print(f"Users loaded: {len(users)}")
        
        if calc_roc == False:
            # Verify structure and show summary
            total_skeletons1 = sum(len(finger_data['fingers']["L"]) for finger_data in users.values())
            total_skeletons2= sum(len(finger_data['fingers']["R"]) for finger_data in users.values())
            print(f"Total skeletons: {total_skeletons1+total_skeletons2}")

            return users
        
        elif calc_roc==True:
            return users
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def save_users_dictionary(users, filename):
    """Save the processed users dictionary"""
    
    # Create directory if it doesn't exist
    os.makedirs("biometric_cache", exist_ok=True)
    
    # Full filepath
    filepath = os.path.join("biometric_cache", filename)
    
    # Save with pickle
    with open(filepath, 'wb') as f:
        pickle.dump(users, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Users dictionary saved successfully!")
    print(f"Location: {filepath}")
    print(f"Users saved: {len(users)}")
    
    return filepath
