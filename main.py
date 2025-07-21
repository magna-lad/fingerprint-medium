import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# takes in the img as the object

# returns an object containing the minutias detail of a fingerprint 
class minutiaLoader:  # will only handle loading of the minutia, segmentation
    def __init__(self,img_path):
        '''
        x_cord,y_cord -> coord of the minutias
        angle-> angle of the minutias
        type-> bifurcation or a ridge
        '''
        self.img_path = img_path
        self.block = 13

        self.img= self.load()
        self.normalised_img= self.normalise(self.img)
        self.segmented_img, self.norm_img, self.mask = self.segmentation(self.normalised_img)

        extractor = minutiaExtractor(self.normalised_img,self.segmented_img,self.norm_img,self.mask,self.block)

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
        segmented_img = img.copy()
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
        segmented_img *= mask
        im = self.normalise(img)

        mean_val = np.mean(im[mask==0])
        std_val = np.std(im[mask==0])
        norm_img = (im - mean_val)/(std_val)
    
        return segmented_img, norm_img, mask


# will output the x,y,type and angle of the point minutia
class minutiaExtractor:
    def __init__(self,normalised_img,segmented_img, norm_img, mask,block):
        self.x_cord = None
        self.y_cord = None
        self.angle = None
        self.type = None
        self.block = block
        
        self.normalised_img = normalised_img
        self.segmented_img=segmented_img
        self.norm_img=norm_img
        self.mask = mask

    # for gabor filters
    def angleCalculation(self,smooth=False):
        j1 = lambda x, y: 2 * x * y
        j2 = lambda x, y: x ** 2 - y ** 2
        j3 = lambda x, y: x ** 2 + y ** 2

        (y, x) = self.normalised_img.shape

        sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

        ySobel = np.array(sobelOperator).astype(np.int_)
        xSobel = np.transpose(ySobel).astype(np.int_)

        result = [[] for i in range(1, y, self.block)]

        # gradients
        # /125 -> to scale down the pixels for better numerical processing
        # *125 -> 'unscale' them
        Gx_ = cv2.filter2D(self.normalised_img/125,-1, ySobel)*125
        Gy_ = cv2.filter2D(self.normalised_img/125,-1, xSobel)*125

        for j in range(1, y, self.block):
            for i in range(1, x, self.block):
                # entering the img via pixel
                nominator = 0
                denominator = 0
                # convolving the block of the image with sobel operators
                for l in range(j, min(j + self.block, y - 1)):
                    for k in range(i, min(i + self.block , x - 1)):
                        Gx = round(Gx_[l, k])  # horizontal gradients at l, k
                        Gy = round(Gy_[l, k])  # vertial gradients at l, k
                        nominator += j1(Gx, Gy)
                        denominator += j2(Gx, Gy)

                if nominator or denominator:
                    angle = (math.pi + math.atan2(nominator, denominator)) / 2
                    result[int((j-1) // self.block)].append(angle)
                else:
                    result[int((j-1) // self.block)].append(0)

        result = np.array(result)

        if smooth:
            result = minutiaExtractor.smooth_angles(result)

        return result
    @staticmethod
    def gauss(x, y):
        sigma_ = 1.0
        return (1 / (2 * math.pi * sigma_)) * math.exp(-(x * x + y * y) / (2 * sigma_))
    @staticmethod
    def kernel_from_function(size, f):
        kernel = [[] for i in range(0, size)]
        for i in range(0, size):
            for j in range(0, size):
                kernel[i].append(f(i - size / 2, j - size / 2))
        return kernel
    @staticmethod
    def smooth_angles(angles):
        angles = np.array(angles)
        cos_angles = np.cos(angles.copy()*2)
        sin_angles = np.sin(angles.copy()*2)

        kernel = np.array(minutiaExtractor.kernel_from_function(5, minutiaExtractor.gauss))

        cos_angles = cv2.filter2D(cos_angles/125,-1, kernel)*125
        sin_angles = cv2.filter2D(sin_angles/125,-1, kernel)*125
        smooth_angles = np.arctan2(sin_angles, cos_angles)/2

        return smooth_angles    
    
    
    



fingerprint = minutiaLoader(r"C:\Users\kound\OneDrive\Desktop\finger-50classes\004\L\004_L3_4.bmp")


plt.figure(figsize=(9, 3))

plt.subplot(1,3,1)
plt.imshow(fingerprint.segmented_img,cmap="gray")

plt.subplot(1,3,2)
plt.imshow(fingerprint.norm_img,cmap="gray")


plt.subplot(1,3,3)
plt.imshow(fingerprint.mask,cmap="gray")
plt.show()

plt.tight_layout()




