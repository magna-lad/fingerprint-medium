import os
import random
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
import warnings
from PIL import Image
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== Fingerprint pipeline ================

def normalise(img):
    return (img - np.mean(img))/(np.std(img))

def create_segmented_and_variance_images(im, w, threshold=.2):
    """
    Returns mask identifying the ROI. Calculates the standard deviation in each image block and threshold the ROI
    It also normalises the intesity values of
    the image so that the ridge regions have zero mean, unit standard
    deviation.
    :param im: Image
    :param w: size of the block
    :param threshold: std threshold
    :return: segmented_image
    """
    (y, x) = im.shape
    threshold = np.std(im)*threshold

    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # apply threshold
    mask[image_variance < threshold] = 0

    # smooth mask with a open/close morphological filter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(w*2, w*2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # normalise segmented image
    segmented_image *= mask
    im = normalise(im)
    mean_val = np.mean(im[mask==0])
    std_val = np.std(im[mask==0])
    norm_img = (im - mean_val)/(std_val)

    return segmented_image, norm_img, mask

def calculate_angles(im, W, smoth=False):
    """
    anisotropy orientation estimate, based on equations 5 from:
    https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf
    :param im:
    :param W: int width of the ridge
    :return: array
    """
    j1 = lambda x, y: 2 * x * y
    j2 = lambda x, y: x ** 2 - y ** 2
    j3 = lambda x, y: x ** 2 + y ** 2

    (y, x) = im.shape

    sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    ySobel = np.array(sobelOperator).astype(np.int_)
    xSobel = np.transpose(ySobel).astype(np.int_)

    result = [[] for i in range(1, y, W)]

    Gx_ = cv2.filter2D(im/125,-1, ySobel)*125
    Gy_ = cv2.filter2D(im/125,-1, xSobel)*125

    for j in range(1, y, W):
        for i in range(1, x, W):
            nominator = 0
            denominator = 0
            for l in range(j, min(j + W, y - 1)):
                for k in range(i, min(i + W , x - 1)):
                    Gx = round(Gx_[l, k])  # horizontal gradients at l, k
                    Gy = round(Gy_[l, k])  # vertial gradients at l, k
                    nominator += j1(Gx, Gy)
                    denominator += j2(Gx, Gy)

            if nominator or denominator:
                angle = (math.pi + math.atan2(nominator, denominator)) / 2
                orientation = np.pi/2 + math.atan2(nominator,denominator)/2
                result[int((j-1) // W)].append(angle)
            else:
                result[int((j-1) // W)].append(0)

    result = np.array(result)

    if smoth:
        result = smooth_angles(result)

    return result

def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))

def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel

def smooth_angles(angles):
    """
    reference: https://airccj.org/CSCP/vol7/csit76809.pdf pg91
    Practically, it is possible to have a block so noisy that the directional estimate is completely false.
    This then causes a very large angular variation between two adjacent blocks. However, a
    fingerprint has some directional continuity, such a variation between two adjacent blocks is then
    representative of a bad estimate. To eliminate such discontinuities, a low-pass filter is applied to
    the directional board.
    :param angles:
    :return:
    """
    angles = np.array(angles)
    cos_angles = np.cos(angles.copy()*2)
    sin_angles = np.sin(angles.copy()*2)

    kernel = np.array(kernel_from_function(5, gauss))

    cos_angles = cv2.filter2D(cos_angles/125,-1, kernel)*125
    sin_angles = cv2.filter2D(sin_angles/125,-1, kernel)*125
    smooth_angles = np.arctan2(sin_angles, cos_angles)/2

    return smooth_angles

def get_line_ends(i, j, W, tang):
    if -1 <= tang and tang <= 1:
        begin = (i, int((-W/2) * tang + j + W/2))
        end = (i + W, int((W/2) * tang + j + W/2))
    else:
        begin = (int(i + W/2 + W/(2 * tang)), j + W//2)
        end = (int(i + W/2 - W/(2 * tang)), j - W//2)
    return (begin, end)

def visualize_angles(im, mask, angles, W):
    (y, x) = im.shape
    result = cv2.cvtColor(np.zeros(im.shape, np.uint8), cv2.COLOR_GRAY2RGB)
    mask_threshold = (W-1)**2
    for i in range(1, x, W):
        for j in range(1, y, W):
            radian = np.sum(mask[j - 1:j + W, i-1:i+W])
            if radian > mask_threshold:
                tang = math.tan(angles[(j - 1) // W][(i - 1) // W])
                (begin, end) = get_line_ends(i, j, W, tang)
                cv2.line(result, begin, end, color=150)

    cv2.resize(result, im.shape, result)
    return result

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
    ridge_noise = np.abs(dilation - ridge_sum); peak_thresh = 2;
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

def ridge_freq(im, mask, orient, block_size, kernel_size, minWaveLength, maxWaveLength):
    # Function to estimate the fingerprint ridge frequency across a
    # fingerprint image.
    rows,cols = im.shape
    freq = np.zeros((rows,cols))

    for row in range(0, rows - block_size, block_size):
        for col in range(0, cols - block_size, block_size):
            image_block = im[row:row + block_size][:, col:col + block_size]
            angle_block = orient[row // block_size][col // block_size]
            if angle_block:
                freq[row:row + block_size][:, col:col + block_size] = frequest(image_block, angle_block, kernel_size,
                                                                               minWaveLength, maxWaveLength)

    freq = freq*mask
    freq_1d = np.reshape(freq,(1,rows*cols))
    ind = np.where(freq_1d>0)
    ind = np.array(ind)
    ind = ind[1,:]
    non_zero_elems_in_freq = freq_1d[0][ind]
    medianfreq = np.median(non_zero_elems_in_freq) * mask
    
    return medianfreq

def gabor_filter(im, orient, freq, kx=0.65, ky=0.65):
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
    im = np.double(im)
    rows, cols = im.shape
    return_img = np.zeros((rows,cols))

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    freq_1d = freq.flatten()
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
    orientindex = np.round(orient/np.pi*180/angleInc)
    for i in range(0,rows//16):
        for j in range(0,cols//16):
            if(orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if(orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex

    # Find indices of matrix points greater than maxsze from the image boundary
    block_size = int(block_size)
    valid_row, valid_col = np.where(freq>0)
    finalind = \
        np.where((valid_row>block_size) & (valid_row<rows - block_size) & (valid_col>block_size) & (valid_col<cols - block_size))

    for k in range(0, np.shape(finalind)[1]):
        r = valid_row[finalind[0][k]]; c = valid_col[finalind[0][k]]
        img_block = im[r-block_size:r+block_size + 1][:,c-block_size:c+block_size + 1]
        return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r//16][c//16]) - 1])

    gabor_img = 255 - np.array((return_img < 0)*255).astype(np.uint8)

    return gabor_img

def skeletonize(image_input):
    """
    https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
    Skeletonization reduces binary objects to 1 pixel wide representations.
    skeletonize works by making successive passes of the image. On each pass, border pixels are identified
    and removed on the condition that they do not break the connectivity of the corresponding object.
    :param image_input: 2d array uint8
    :return:
    """
    image = np.zeros_like(image_input)
    image[image_input == 0] = 1.0
    output = np.zeros_like(image_input)

    skeleton = skelt(image)

    output[skeleton] = 255
    cv2.bitwise_not(output, output)

    return output

def thinning_morph(image, kernel):
    """
    Thinning image using morphological operations
    :param image: 2d array uint8
    :param kernel: 3x3 2d array unint8
    :return: thin images
    """
    thining_image = np.zeros_like(image)
    img = image.copy()

    while 1:
        erosion = cv2.erode(img, kernel, iterations = 1)
        dilatate = cv2.dilate(erosion, kernel, iterations = 1)

        subs_img = np.subtract(img, dilatate)
        cv2.bitwise_or(thining_image, subs_img, thining_image)
        img = erosion.copy()

        done = (np.sum(img) == 0)

        if done:
          break

    # shift down and compare one pixel offset
    down = np.zeros_like(thining_image)
    down[1:-1, :] = thining_image[0:-2, ]
    down_mask = np.subtract(down, thining_image)
    down_mask[0:-2, :] = down_mask[1:-1, ]
    cv2.imshow('down', down_mask)

    # shift right and compare one pixel offset
    left = np.zeros_like(thining_image)
    left[:, 1:-1] = thining_image[:, 0:-2]
    left_mask = np.subtract(left, thining_image)
    left_mask[:, 0:-2] = left_mask[:, 1:-1]
    cv2.imshow('left', left_mask)

    # combine left and down mask
    cv2.bitwise_or(down_mask, down_mask, thining_image)
    output = np.zeros_like(thining_image)
    output[thining_image < 250] = 255

    return output

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

def calculate_minutiaes(im, kernel_size=3):
    biniry_image = np.zeros_like(im)
    biniry_image[im<10] = 1.0
    biniry_image = biniry_image.astype(np.int8)

    (y, x) = im.shape
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)}

    # iterate each pixel minutia
    for i in range(1, x - kernel_size//2):
        for j in range(1, y - kernel_size//2):
            minutiae = minutiae_at(biniry_image, j, i, kernel_size)
            if minutiae != "none":
                cv2.circle(result, (i,j), radius=2, color=colors[minutiae], thickness=2)

    
    return result

def fingerprint_pipeline(input_img):
    block_size = 16

    # pipe line picture re https://www.cse.iitk.ac.in/users/biometrics/pages/111.JPG
    # normalization -> orientation -> frequency -> mask -> filtering

    # normalization - removes the effects of sensor noise and finger pressure differences.
    normalised_img = normalise(input_img.copy())
    #print('normalised')
    # ROI and normalisation
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalised_img, block_size, 0.2)

    # orientations
    angles = calculate_angles(normalised_img, W=block_size, smoth=False)
    orientation_img = visualize_angles(segmented_img, mask, angles, W=block_size)
    #print('angles found')
    # find the overall frequency of ridges in Wavelet Domain
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15)
    #print('freq found')
    # create gabor filter and do the actual filtering
    gabor_img = gabor_filter(normim, angles, freq)
    #print('gabor_img found')
    # thinning oor skeletonize
    thin_image = skeletonize(gabor_img)
    #print('thin image found')
    # minutias
    minutias = calculate_minutiaes(thin_image)
    #print('minutias found')
    #print(minutias)
    return minutias

# ========== UTILITIES ==========

def create_frequency_distribution_curve(data, num_classes=20, smooth=True):
    if len(data) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    data_range = np.max(data) - np.min(data)
    if data_range == 0:
        return np.array([np.mean(data)]), np.array([1.0]), np.array([np.mean(data)]), np.array([1.0])
    bins = np.linspace(np.min(data), np.max(data), num_classes + 1)
    frequencies, bin_edges = np.histogram(data, bins=bins, density=True)
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    if smooth and len(midpoints) > 3:
        f = interp1d(midpoints, frequencies, kind='cubic', bounds_error=False, fill_value=0)
        x_smooth = np.linspace(midpoints[0], midpoints[-1], 300)
        y_smooth = f(x_smooth)
        return x_smooth, y_smooth, midpoints, frequencies
    else:
        return midpoints, frequencies, midpoints, frequencies

def plot_frequency_distributions(genuine_scores, impostor_scores, title="Score Distribution"):
    if len(genuine_scores) > 0:
        x_gen, y_gen, _, _ = create_frequency_distribution_curve(genuine_scores)
        plt.plot(x_gen, y_gen, 'b-', linewidth=2, label='Genuine Scores', alpha=0.8)
        plt.fill_between(x_gen, y_gen, alpha=0.3, color='blue')
        plt.hist(genuine_scores, bins=20, alpha=0.4, density=True, color='blue', edgecolor='black', linewidth=0.5)
    if len(impostor_scores) > 0:
        x_imp, y_imp, _, _ = create_frequency_distribution_curve(impostor_scores)
        plt.plot(x_imp, y_imp, 'r-', linewidth=2, label='Impostor Scores', alpha=0.8)
        plt.fill_between(x_imp, y_imp, alpha=0.3, color='red')
        plt.hist(impostor_scores, bins=20, alpha=0.4, density=True, color='red', edgecolor='black', linewidth=0.5)
    plt.xlabel('Score Value')
    plt.ylabel('Frequency Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

def analyze_distribution_characteristics(scores, labels):
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return {}
    analysis = {
        'genuine': {
            'mean': np.mean(genuine_scores),
            'std': np.std(genuine_scores),
            'median': np.median(genuine_scores),
            'skewness': stats.skew(genuine_scores),
            'kurtosis': stats.kurtosis(genuine_scores),
            'min': np.min(genuine_scores),
            'max': np.max(genuine_scores)
        },
        'impostor': {
            'mean': np.mean(impostor_scores),
            'std': np.std(impostor_scores),
            'median': np.median(impostor_scores),
            'skewness': stats.skew(impostor_scores),
            'kurtosis': stats.kurtosis(impostor_scores),
            'min': np.min(impostor_scores),
            'max': np.max(impostor_scores)
        },
        'separation': {
            'mean_difference': np.mean(genuine_scores) - np.mean(impostor_scores),
            'decidability_index': calculate_decidability_index(genuine_scores, impostor_scores)
        }
    }
    return analysis

def compute_brisque_quality(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = gray.astype(np.float64)
    mu = cv2.GaussianBlur(gray, (7, 7), 1.166)
    mu_sq = cv2.GaussianBlur(gray**2, (7, 7), 1.166)
    sigma = np.sqrt(np.abs(mu_sq - mu**2))
    mscn = (gray - mu) / (sigma + 1)
    features = [np.mean(mscn), np.var(mscn), np.mean(np.abs(mscn)), np.mean(mscn**2)]
    shifts = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for shift in shifts:
        shifted = np.roll(np.roll(mscn, shift[0], axis=0), shift[1], axis=1)
        product = mscn * shifted
        features.extend([np.mean(product), np.var(product), np.mean(np.abs(product))])
    fvec = np.array(features)
    fvec = (fvec - np.mean(fvec)) / (np.std(fvec) + 1e-8)
    return np.clip(50 + 25 * np.tanh(np.mean(fvec)), 0, 100)

def compute_reliability_factor(beta):
    return 1.0 - (beta / 100.0)

def atch_score_fingerprint(img1, img2):
    img1=Image.fromarray(img1)
    img2=Image.fromarray(img2)
    minutias1 = fingerprint_pipeline(img1)  # Assume returns a 1D numpy array
    #print(minutias1)
    minutias2 = fingerprint_pipeline(img2)  # Assume returns a 1D numpy array
    # Ensure arrays are 1D and of equal length
    #print(minutias2)
    #minutias1 = np.array(minutias1)
    #minutias2 = np.array(minutias2)
    # all good till here
    minutias1=minutias1.flatten()
    minutias2=minutias2.flatten()
    if len(minutias1) != len(minutias2):
        raise ValueError("Minutiae arrays must be of equal length for Pearson correlation.")

    print('started pearson')
    
    # Apply Pearson correlation
    correlation, p_value = pearsonr(minutias1, minutias2)
    #print(correlation)
    # The score can be the absolute correlation (0 to 1 for similarity)
    score = abs(correlation)  # Use absolute value for positive similarity measure
    print(min(1.0,score)) # error is here
    
    return min(1.0,score)



from scipy.stats import pearsonr
import numpy as np
from PIL import Image

def match_score_fingerprint(img1, img2):
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)

    minutias1 = fingerprint_pipeline(img1)  # Assume returns 1D array or flat features
    minutias2 = fingerprint_pipeline(img2)

    # Convert and flatten safely
    minutias1 = np.array(minutias1).flatten()
    minutias2 = np.array(minutias2).flatten()

    # Empty or invalid array check
    if minutias1.size == 0 or minutias2.size == 0:
        print("Empty feature array returned.")
        return 0.0

    # Check equal lengths
    if minutias1.shape != minutias2.shape:
        print("Feature arrays of unequal shape.")
        return 0.0

    # Pearson correlation must not operate on constant arrays
    if np.all(minutias1 == minutias1[0]) or np.all(minutias2 == minutias2[0]):
        print("Constant array — Pearson correlation undefined.")
        return 0.0

    try:
        print("Starting Pearson correlation...")
        correlation, p_value = pearsonr(minutias1, minutias2)
        score = abs(correlation)
        print(f"Score: {min(1.0, score)}")
        return min(1.0, score)
    except Exception as e:
        print(f"Pearson correlation failed: {e}")
        return 0.0

def adaptive_score_fusion(scores, betas, tau):
    scores = np.array(scores)
    betas = np.array(betas)
    alphas = np.array([compute_reliability_factor(b) for b in betas])
    omegas = alphas * scores
    phis = omegas - np.sqrt(np.maximum(0, tau**2 - omegas**2))
    lambdas = np.abs(phis - tau)
    xi = np.sum(lambdas * phis)
    return np.mean(phis) + xi / len(scores), alphas, omegas, phis, lambdas, xi

def calculate_decidability_index(genuine, impostor):
    if len(genuine) == 0 or len(impostor) == 0:
        return 0.0
    mu_g, mu_i = np.mean(genuine), np.mean(impostor)
    std_g, std_i = np.std(genuine), np.std(impostor)
    if (std_g**2 + std_i**2) == 0:
        return 0.0
    return abs(mu_g - mu_i) / np.sqrt(0.5 * (std_g**2 + std_i**2))

def oad_users(data_dir):
    users = {}
    for uid in os.listdir(data_dir):
        path = os.path.join(data_dir, uid)
        if not os.path.isdir(path):
            continue
        imgs = []
        d = os.path.join(path, 'Fingerprint')
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                    img = cv2.imread(os.path.join(d, f),0)
                    if img is not None:
                        imgs.append(cv2.resize(img, (96, 96)))
        users[uid] = {'finger': imgs}
    return users



def oad_users(data_dir):
    users = {}
    for uid in os.listdir(data_dir):
        path = os.path.join(data_dir, uid)
        if not os.path.isdir(path):
            continue
        
        finger_data = {'L': [], 'R': []}
        
        for hand in ['L', 'R']:
            hand_dir = os.path.join(path, hand)
            if os.path.isdir(hand_dir):
                for f in os.listdir(hand_dir):
                    if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                        img_path = os.path.join(hand_dir, f)
                        img = cv2.imread(img_path, 0)
                        if img is not None:
                            finger_data[hand].append(cv2.resize(img, (96, 96)))
        
        users[uid] = {'finger': finger_data}
    
    return users


def load_users(data_dir):
    users = {}
    for uid in os.listdir(data_dir):
        uid_path = os.path.join(data_dir, uid)
        if not os.path.isdir(uid_path):
            continue
        
        imgs = []

        for hand in ['L', 'R']:
            hand_dir = os.path.join(uid_path, hand)
            if os.path.isdir(hand_dir):
                for f in os.listdir(hand_dir): # entering the hand folder
                    if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg')):
                        img_path = os.path.join(hand_dir, f)
                        img = cv2.imread(img_path, 0)
                        if img is not None:
                            img = cv2.resize(img, (96, 96))
                            imgs.append(img)

        users[uid] = {'finger': imgs}
    
    return users


def main():
    data_dir = "/kaggle/input/casiav5/finger"
    users = load_users(data_dir)
    #print(users)

    uids = list(users.keys())
    if len(uids) < 2:
        print("Need at least 2 users for evaluation")
        return

    all_scores, all_betas, all_labels = [], [], []

    for probe_id in tqdm(uids, desc='Processing users'):
        #print('start')
        pr = users[probe_id]
        if not pr['finger']:
            continue
        #print('start')
        for gal_id in uids:
            ga = users[gal_id]
            if not ga['finger']:
                continue
            try:
                #print('star00t')
                #print('finger chosen')
                pp = random.choice(pr['finger'])
                #pp=pp.flatten()
                #print(pp)
                pg = random.choice(ga['finger'])
                #pg=pg.flatten()
                
                sp = match_score_fingerprint(pp, pg)
                

                print('brisque quality')
                bp = compute_brisque_quality(pg)
                print('scores')
                all_scores.append([sp])
                print('betas')
                all_betas.append([bp])
                all_labels.append(1 if probe_id == gal_id else 0)
                print(all_labels)
                print('label operation')
            except Exception as e:
                print(f"Error processing {probe_id}-{gal_id}: {e}")
                continue

    if len(all_scores) == 0:
        print("No valid scores generated")
        return

    scores = np.array(all_scores).reshape(-1, 1)
    betas = np.array(all_betas).reshape(-1, 1)
    labels = np.array(all_labels)

    print(f"Generated {len(scores)} score pairs")
    print(f"Genuine pairs: {np.sum(labels)}, Impostor pairs: {np.sum(1-labels)}")

    tau = np.mean(scores[labels == 1]) if np.sum(labels == 1) > 0 else 0.5
    print(f"Optimal threshold τ = {tau:.4f}")

    fused_scores = np.array([
        adaptive_score_fusion([scores[i][0]], [betas[i][0]], tau)[0]
        for i in range(len(scores))
    ])

    if fused_scores.max() > fused_scores.min():
        fused_scores = (fused_scores - fused_scores.min()) / (fused_scores.max() - fused_scores.min())

    fpr, tpr, thresholds = roc_curve(labels, fused_scores)
    roc_auc = auc(fpr, tpr)
    eer_idx = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    predictions = (fused_scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    di = calculate_decidability_index(fused_scores[labels == 1], fused_scores[labels == 0])

    print(f"\n=== RESULTS ===")
    print(f"AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Decidability Index: {di:.4f}")

    analysis = analyze_distribution_characteristics(fused_scores, labels)
    if analysis:
        print(f"\n=== DISTRIBUTION ANALYSIS ===")
        print(f"Genuine - Mean: {analysis['genuine']['mean']:.4f}, Std: {analysis['genuine']['std']:.4f}")
        print(f"Impostor - Mean: {analysis['impostor']['mean']:.4f}, Std: {analysis['impostor']['std']:.4f}")
        print(f"Separation - Mean Diff: {analysis['separation']['mean_difference']:.4f}")

    # Plots
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    axes[0].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(fpr[eer_idx], tpr[eer_idx], color='red', s=100, label=f'EER={eer:.3f}')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True)

    # Score Distribution
    genuine_fused = fused_scores[labels == 1]
    impostor_fused = fused_scores[labels == 0]
    plt.sca(axes[1])
    plot_frequency_distributions(genuine_fused, impostor_fused, "Fused Score Distribution")
#
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
