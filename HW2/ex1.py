import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os


def read_img(img_path):
    """
        Read grayscale image
        Inputs:
        img_path: str: image path
        Returns:
        img: cv2 image
    """
    return cv2.imread(img_path, 0)


def padding_img(img, filter_size=3):
    """
    The surrogate function for the filter functions.
    The goal of the function: replicate padding the image such that when applying the kernel with the size of filter_size, the padded image will be the same size as the original image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter
    Return:
        padded_img: cv2 image: the padding image
    """
    #set up for padding
    height, width = img.shape
    pad_size = filter_size // 2
    padded_img = np.zeros((height + 2 * pad_size, width + 2 * pad_size), dtype=img.dtype)

    #copy the original image to the center of padded image
    padded_img[pad_size:pad_size + height, pad_size:pad_size + width] = img

    #replicate padding at non-corner borders
    padded_img[:pad_size, pad_size:-pad_size] = img[0, :]    #top rows
    padded_img[-pad_size:, pad_size:-pad_size] = img[-1, :]  #bottom rows
    padded_img[pad_size:-pad_size, :pad_size] = img[:, 0].reshape(-1, 1)    #left columns
    padded_img[pad_size:-pad_size, -pad_size:] = img[:, -1].reshape(-1, 1)  #right columns

    #replicate padding at corners
    padded_img[:pad_size, :pad_size] = img[0, 0]            #top-left
    padded_img[:pad_size, -pad_size:] = img[0, -1]         #top-right
    padded_img[-pad_size:, :pad_size] = img[-1, 0]          #bottom-left
    padded_img[-pad_size:, -pad_size:] = img[-1, -1]        #bottom-right

    return padded_img

def mean_filter(img, filter_size=3):
    """
    Smoothing image with mean square filter with the size of filter_size. Use replicate padding for the image.
    WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
    Inputs:
        img: cv2 image: original image
        filter_size: int: size of square filter,
    Return:
        smoothed_img: cv2 image: the smoothed image with mean filter.
    """
    #set up for smoothing
    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros_like(img, dtype=img.dtype)
    height, width = img.shape

    #smoothing
    for i in range(height):
        for j in range(width):
            smoothed_img[i, j] = np.mean(padded_img[i:i + filter_size, j:j + filter_size])

    return smoothed_img


def median_filter(img, filter_size=3):
    """
        Smoothing image with median square filter with the size of filter_size. Use replicate padding for the image.
        WARNING: Do not use the exterior functions from available libraries such as OpenCV, scikit-image, etc. Just do from scratch using function from the numpy library or functions in pure Python.
        Inputs:
            img: cv2 image: original image
            filter_size: int: size of square filter
        Return:
            smoothed_img: cv2 image: the smoothed image with median filter.
    """
    #set up for smoothing
    padded_img = padding_img(img, filter_size)
    smoothed_img = np.zeros_like(img, dtype=img.dtype)
    height, width = img.shape

    #smoothing
    for i in range(height):
        for j in range(width):
            smoothed_img[i, j] = np.median(padded_img[i:i + filter_size, j:j + filter_size])

    return smoothed_img


def psnr(gt_img, smooth_img):
    """
        Calculate the PSNR metric
        Inputs:
            gt_img: cv2 image: groundtruth image
            smooth_img: cv2 image: smoothed image
        Outputs:
            psnr_score: PSNR score
    """
    mse = np.mean((gt_img - smooth_img) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_score = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr_score



def show_res(before_img, after_img):
    """
        Show the original image and the corresponding smooth image
        Inputs:
            before_img: cv2: image before smoothing
            after_img: cv2: corresponding smoothed image
        Return:
            None
    """
    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.imshow(before_img, cmap='gray')
    plt.title('Before')

    plt.subplot(1, 2, 2)
    plt.imshow(after_img, cmap='gray')
    plt.title('After')
    plt.show()


if __name__ == '__main__':
    img_noise = "hw2/ex1_images/noise.png"  #path to the noise image
    img_gt = "hw2/ex1_images/ori_img.png"   #path to the groundtruth image
    img = read_img(img_noise)
    filter_size = 3

    # Mean filter
    mean_smoothed_img = mean_filter(img, filter_size)
    show_res(img, mean_smoothed_img)
    print('PSNR score of mean filter: ', psnr(img, mean_smoothed_img))

    # Median filter
    median_smoothed_img = median_filter(img, filter_size)
    show_res(img, median_smoothed_img)
    print('PSNR score of median filter: ', psnr(img, median_smoothed_img))

