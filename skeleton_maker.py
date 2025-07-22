import numpy as np
import cv2
import torch
import math
import scipy.ndimage
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from skimage.morphology import skeletonize as skelt



# will output the x,y,type and angle of the point minutia
class skeleton_maker:
    def __init__(self,normalised_img,segmented_img, norm_img, mask,block):
        self.x_cord = None
        self.y_cord = None
        self.angle_minutia = None
        self.type = None
        self.gabor_img=None
        self.skeleton = None


        self.block = block
        self.normalised_img = normalised_img
        self.segmented_img=segmented_img
        self.norm_img=norm_img
        self.mask = mask
        self.angle_gabor=[] # required for gabor filters
        self.freq=[]
    
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
            result = skeleton_maker.smooth_angles(result)

        self.angle_gabor = result
        return self.angle_gabor
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

        kernel = np.array(skeleton_maker.kernel_from_function(5, skeleton_maker.gauss))

        cos_angles = cv2.filter2D(cos_angles/125,-1, kernel)*125
        sin_angles = cv2.filter2D(sin_angles/125,-1, kernel)*125
        smooth_angles = np.arctan2(sin_angles, cos_angles)/2

        return smooth_angles    
    

    # just to visualise the angles
    def get_line_ends(self,i, j, tang):
        if -1 <= tang and tang <= 1:
            begin = (i, int((-self.block/2) * tang + j + self.block/2))
            end = (i + self.block, int((self.block/2) * tang + j + self.block/2))
        else:
            begin = (int(i + self.block/2 + self.block/(2 * tang)), j + self.block//2)
            end = (int(i + self.block/2 - self.block/(2 * tang)), j - self.block//2)
        return (begin, end)

    # segmented_img-im
    # mask-mask
    # angle- angle_gabor
    # w- self.block
    def visualize_angles(self):
        (y, x) = self.segmented_img.shape
        result = cv2.cvtColor(np.zeros(self.segmented_img.shape, np.uint8), cv2.COLOR_GRAY2RGB)
        mask_threshold = (self.segmented_img-1)**2
        for i in range(1, x, self.block):
            for j in range(1, y, self.block):
                radian = np.sum(self.mask[j - 1:j + self.block, i-1:i+self.block])
                if radian > mask_threshold:
                    tang = math.tan(self.angle_gabor[(j - 1) // self.block][(i - 1) // self.block])
                    (begin, end) = skeleton_maker.get_line_ends(i, j, tang)
                    cv2.line(result, begin, end, color=150)

        cv2.resize(result, self.segmented_img.shape, result)
        return result
    


    # for gabor filters
                #self.normim,sel.mask,self.angle_gabor,self.block
    #freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    def ridge_freq(self,kernel_size=5, minWaveLength=5, maxWaveLength=15):
        # Function to estimate the fingerprint ridge frequency across a
    # fingerprint image.
        rows,cols = self.norm_img.shape
        freq = np.zeros((rows,cols))

        for row in range(0, rows - self.block, self.block):
            for col in range(0, cols - self.block, self.block):
                image_block = self.norm_img[row:row + self.block][:, col:col + self.block]
                angle_block = self.angle_gabor[row // self.block][col // self.block]
                if angle_block:
                    freq[row:row + self.block][:, col:col + self.block] = skeleton_maker.frequest(image_block, angle_block, kernel_size,minWaveLength, maxWaveLength)

        freq = freq*self.mask
        freq_1d = np.reshape(freq,(1,rows*cols))
        ind = np.where(freq_1d>0)
        ind = np.array(ind)
        ind = ind[1,:]
        non_zero_elems_in_freq = freq_1d[0][ind]
        medianfreq = np.median(non_zero_elems_in_freq) * self.mask
        self.freq=medianfreq
        return self.freq
    
    @staticmethod
    def frequest(im, orientim, kernel_size, minWaveLength, maxWaveLength):
        """
        Based on https://pdfs.semanticscholar.org/ca0d/a7c552877e30e1c5d87dfcfb8b5972b0acd9.pdf pg.14
        Function to estimate the fingerprint ridge frequency within a small block
        of a fingerprint image.
        An image block the same size as im with all values set to the estimated ridge spatial frequency.  If a
        ridge frequency cannot be found, or cannot be found within the limits set by min and max Wavlength freqim is set to zeros.
        """
        rows, cols = np.shape(im)

        # Find mean orientation within the block. This is done by averaging the
        # sines and cosines of the doubled angles before reconstructing the angle again.
        cosorient = np.cos(2*orientim) # np.mean(np.cos(2*orientim))
        sinorient = np.sin(2*orientim) # np.mean(np.sin(2*orientim))
        block_orient = math.atan2(sinorient,cosorient)/2

        # Rotate the image block so that the ridges are vertical
        rotim = scipy.ndimage.rotate(im,block_orient/np.pi*180 + 90,axes=(1,0),reshape = False,order = 3,mode = 'nearest')

        # Now crop the image so that the rotated image does not contain any invalid regions.
        cropsze = int(np.fix(rows/np.sqrt(2)))
        offset = int(np.fix((rows-cropsze)/2))
        rotim = rotim[offset:offset+cropsze][:,offset:offset+cropsze]

        # Sum down the columns to get a projection of the grey values down the ridges.
        ridge_sum = np.sum(rotim, axis = 0)
        dilation = scipy.ndimage.grey_dilation(ridge_sum, kernel_size, structure=np.ones(kernel_size))
        ridge_noise = np.abs(dilation - ridge_sum); peak_thresh = 2
        maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
        maxind = np.where(maxpts)
        _, no_of_peaks = np.shape(maxind)

        # Determine the spatial frequency of the ridges by dividing the
        # distance between the 1st and last peaks by the (No of peaks-1). If no
        # peaks are detected, or the wavelength is outside the allowed bounds, the frequency image is set to 0
        if(no_of_peaks<2):
            freq_block = np.zeros(im.shape)
        else:
            waveLength = (maxind[0][-1] - maxind[0][0])/(no_of_peaks - 1)
            if waveLength>=minWaveLength and waveLength<=maxWaveLength:
                freq_block = 1/np.double(waveLength) * np.ones(im.shape)
            else:
                freq_block = np.zeros(im.shape)
        return(freq_block)

                    #norm_img, angle_gabor, freq
    def gabor_filter(self,kx=0.65, ky=0.65):
        """
        Gabor filter is a linear filter used for edge detection. Gabor filter can be viewed as a sinusoidal plane of
        particular frequency and orientation, modulated by a Gaussian envelope.
        :param im:
        :param orient:
        :param freq:
        :param kx:
        :param ky:
        :return:
        """
        angleInc = 3
        im = np.double(self.norm_img)
        rows, cols = im.shape
        return_img = np.zeros((rows,cols))

        # Round the array of frequencies to the nearest 0.01 to reduce the
        # number of distinct frequencies we have to deal with.
        freq_1d = self.freq.flatten()
        frequency_ind = np.array(np.where(freq_1d>0))
        non_zero_elems_in_freq = freq_1d[frequency_ind]
        non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq*100)))/100
        unfreq = np.unique(non_zero_elems_in_freq)

        # Generate filters corresponding to these distinct frequencies and
        # orientations in 'angleInc' increments.
        sigma_x = 1/unfreq*kx
        sigma_y = 1/unfreq*ky
        block_size = np.round(3*np.max([sigma_x,sigma_y]))
        block_size = int(block_size)
        array = np.linspace(-block_size,block_size,(2*block_size + 1))
        x, y = np.meshgrid(array, array)

        # gabor filter equation
        reffilter = np.exp(-(((np.power(x,2))/(sigma_x*sigma_x) + (np.power(y,2))/(sigma_y*sigma_y)))) * np.cos(2*np.pi*unfreq[0]*x)
        filt_rows, filt_cols = reffilter.shape
        gabor_filter = np.array(np.zeros((180//angleInc, filt_rows, filt_cols)))

        # Generate rotated versions of the filter.
        for degree in range(0,180//angleInc):
            rot_filt = scipy.ndimage.rotate(reffilter,-(degree*angleInc + 90),reshape = False)
            gabor_filter[degree] = rot_filt

        # Convert orientation matrix values from radians to an index value that corresponds to round(degrees/angleInc)
        maxorientindex = np.round(180/angleInc)
        orientindex = np.round(self.angle_gabor/np.pi*180/angleInc)
        for i in range(0,rows//16):
            for j in range(0,cols//16):
                if(orientindex[i][j] < 1):
                    orientindex[i][j] = orientindex[i][j] + maxorientindex
                if(orientindex[i][j] > maxorientindex):
                    orientindex[i][j] = orientindex[i][j] - maxorientindex

        # Find indices of matrix points greater than maxsze from the image boundary
        block_size = int(block_size)
        valid_row, valid_col = np.where(self.freq>0)
        finalind = \
            np.where((valid_row>block_size) & (valid_row<rows - block_size) & (valid_col>block_size) & (valid_col<cols - block_size))

        for k in range(0, np.shape(finalind)[1]):
            r = valid_row[finalind[0][k]]; c = valid_col[finalind[0][k]]
            img_block = im[r-block_size:r+block_size + 1][:,c-block_size:c+block_size + 1]
            return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r//16][c//16]) - 1])

        gabor_img = 255 - np.array((return_img < 0)*255).astype(np.uint8)
        self.gabor_img = gabor_img
        return gabor_img
    
    def skeletonize(self):
        """
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
        Skeletonization reduces binary objects to 1 pixel wide representations.
        skeletonize works by making successive passes of the image. On each pass, border pixels are identified
        and removed on the condition that they do not break the connectivity of the corresponding object.
        :param image_input: 2d array uint8
        :return:
        """
        image = np.zeros_like(self.gabor_img)
        image[self.gabor_img == 0] = 1.0
        output = np.zeros_like(self.gabor_img)

        skeleton = skelt(image)

        output[skeleton] = 255
        cv2.bitwise_not(output, output)
        self.skeleton = output
        return output

    def calculate_minutiaes(self, kernel_size=3):
        biniry_image = np.zeros_like(self.skeleton)
        biniry_image[self.skeleton<10] = 1.0
        biniry_image = biniry_image.astype(np.int8)

        (y, x) = self.skeleton.shape
        result = cv2.cvtColor(self.gabor_img, cv2.COLOR_GRAY2RGB)
        colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}

        # iterate each pixel minutia
        for i in range(1, x - kernel_size//2):
            for j in range(1, y - kernel_size//2):
                minutiae = skeleton_maker.minutiae_at(biniry_image, j, i, kernel_size)
                if minutiae != "none":
                        cv2.circle(result, (i,j), radius=2, color=colors[minutiae], thickness=2)
            return result
    
    @staticmethod
    def minutiae_at(pixels, i, j, kernel_size):
        """
        https://airccj.org/CSCP/vol7/csit76809.pdf pg93
        Crossing number methods is a really simple way to detect ridge endings and ridge bifurcations.
        Then the crossing number algorithm will look at 3x3 pixel blocks:

        if middle pixel is black (represents ridge):
        if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
        if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation

        :param pixels:
        :param i:
        :param j:
        :return:
        """
        # if middle pixel is black (represents ridge)
        if pixels[i][j] == 1:

            if kernel_size == 3:
                cells = [(-1, -1), (-1, 0), (-1, 1),        # p1 p2 p3
                       (0, 1),  (1, 1),  (1, 0),            # p8    p4
                      (1, -1), (0, -1), (-1, -1)]           # p7 p6 p5
            else:
                cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),                 # p1 p2   p3
                       (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),               # p8      p4
                      (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]           # p7 p6   p5

            values = [pixels[i + l][j + k] for k, l in cells]

            # count crossing how many times it goes from 0 to 1
            crossings = 0
            for k in range(0, len(values)-1):
                crossings += abs(values[k] - values[k + 1])
            crossings //= 2

            # if pixel on boundary are crossed with the ridge once, then it is a possible ridge ending
            # if pixel on boundary are crossed with the ridge three times, then it is a ridge bifurcation
            if crossings == 1:
                return "ending"
            if crossings == 3:
                return "bifurcation"

        return "none"
    

    def fingerprintPipeline(self):
       gabor_angles=self.angleCalculation()
       freq = self.ridge_freq()
       gbr_img = self.gabor_filter()
       img_thin = self.skeletonize()
       plt.imshow(img_thin)
       return  gabor_angles, freq, gbr_img, img_thin


