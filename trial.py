import pickle

# Path to your pickle file
file_path = r"C:\Users\kound\OneDrive\Desktop\fingerprint_mine\biometric_cache\processed_skeletons.pkl"

# Load the contents
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# Now you can inspect the structure
print(type(data))               # e.g. dict
print(len(data))                # Number of users

# Example: Print first user ID and its data
for user_id, user_data in data.items():
    print("User ID:", user_id)
    print("Keys:", user_data.keys())  # Should include 'finger', maybe 'minutiae'
    
    print(f"Number of fingerprint images: {len(user_data['finger'])}")
    
    # If minutiae exists
    if 'minutiae' in user_data:
        print(f"Number of minutiae lists: {len(user_data['minutiae'])}")
    
    break  # Remove this if you want to loop over all users
