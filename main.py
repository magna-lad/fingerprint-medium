import matplotlib.pyplot as plt
from minutia_loader import minutiaLoader
from skeleton_maker import skeleton_maker
from reader import load_users
# takes in the img as the object

# returns an object containing the minutias detail of a fingerprint 
#fingerprint = minutiaLoader(r"C:\Users\kound\OneDrive\Desktop\finger-50classes\044\R\044_R2_0.bmp")

#normalised_img,segmented_img, norm_img, mask,block

from tqdm import tqdm                          # ➊ make sure tqdm is imported

def main():
    data_dir = r"C:\Users\kound\OneDrive\Desktop\finger"
    users = load_users(data_dir)               # paths to all images
    #print(users)
    skeleton = []

    # tqdm wraps the iterable; desc shows a label on the bar
    for user in tqdm(users, desc="Processing users"):
        fingerprint      = minutiaLoader(user)
        skeleton_image   = skeleton_maker(
            fingerprint.normalised_img,
            fingerprint.segmented_img,
            fingerprint.norm_img,
            fingerprint.mask,
            fingerprint.block
        )
        skeleton_image.fingerprintPipeline()
        skeleton.append(skeleton_image.skeleton)

    print(f"Total skeletons generated: {len(skeleton)}")

    plt.imshow(skeleton[0])
    plt.show()


    

if __name__ == '__main__':
    main()
