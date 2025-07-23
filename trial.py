import pickle
import os
import matplotlib.pyplot as plt

def load_users_dictionary(filename="processed_skeletons.pkl"):
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
        
        # Verify structure and show summary
        total_skeletons = sum(len(finger_data['finger']) for finger_data in users.values())
        print(f"Total skeletons: {total_skeletons}")
        
        return users
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
users = load_users_dictionary()
plt.imshow(users['009']['finger'][0])
plt.show()