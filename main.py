import cv2
import numpy as np
import matplotlib.pyplot as plt

# takes in the img as the object

# returns an object containing the minutias detail of a fingerprint 
class minutias:
    def __init__(self,img_path):
        '''
        x_cord,y_cord -> coord of the minutias
        angle-> angle of the minutias
        type-> bifurcation or a ridge
        '''
        self.img_path = img_path
        self.x_cord = None
        self.y_cord = None
        self.angle = None
        self.type = None
        self.block = 16

        self.img= self.load()
        self.normalised_img= self.normalise(self.img)
        self.segmented_img, self.norm_img, self.mask = self.segmentation(self.normalised_img)

    def load(self):
        img = cv2.imread(self.img_path,0) # image is already loaded in grayscale
        if img is None:
            raise ValueError(f"Could not load image from {self.img_path}")
        img = cv2.resize(img,(96,96))
        return img
    
    # normalise the image

    def normalise(self,img):
        return (img - np.mean(img))/(np.std(img))
    
    # segmenting the image to filter out the ROI

    def segmentation(self,img,threshold=0.2 ):
        (h,w) = img.shape #r,c
        threshold = np.std(img)*threshold

        image_variance = np.zeros(img.shape)
        segmented_image = img.copy()
        mask = np.ones_like(img)

        # traversing the image
        for i in range(0,w,self.block):
            for j in range(0,h,self.block):
                box = [i,j,min(i+self.block,w),min(j+self.block,h)]
                    # [start_col,start_row,end_col,end_row]
                block_std = np.std(img[box[1]:box[3],box[0]:box[2]]) # block whose std dev is need to be found out

                image_variance[box[1]:box[3], box[0]:box[2]] = block_std

        mask[image_variance < threshold] = 0

        # smooth mask with a open/close morphological filter
        kernel_size = min(self.block * 1.5, w // 5)  # Cap kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
        # normalise segmented image
        segmented_image *= mask
        im = self.normalise(img)

        mean_val = np.mean(im[mask==0])
        std_val = np.std(im[mask==0])
        norm_img = (im - mean_val)/(std_val)
    
        return segmented_image, norm_img, mask
    

fingerprint = minutias(r"C:\Users\kound\OneDrive\Desktop\finger-50classes\004\L\004_L3_4.bmp")

plt.imshow(fingerprint.norm_img,cmap="gray")
plt.show()






