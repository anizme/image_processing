import io
import ipywidgets as widgets
import matplotlib.pyplot as plt
import PIL.Image
import numpy as np
import urllib
from skimage.transform import resize
from matplotlib.image import imread
import os
from IPython.display import display
from skimage import io as io_url
import cv2
import numpy as np
from PIL import Image

def read_img(img_path, img_size=(512, 512)):
  """
    + Đọc ảnh
    + Chuyển thành grayscale
    + Thay đổi kích thước ảnh thành img_size
  """
  img = cv2.imread(img_path, 0)
  grayscale_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  grayscale_img = cv2.resize(grayscale_img, img_size)
  return grayscale_img


def create_hybrid_img(img1, img2, r):
  """
  Create hydrid image
  Params:
    img1: numpy image 1
    img2: numpy image 2
    r: radius that defines the filled circle of frequency of image 1. Refer to the homework title to know more.
  """
  h, w = img1.shape

  img1_fft = np.fft.fft2(img1)
  img2_fft = np.fft.fft2(img2)

  img1_ffshift = np.fft.fftshift(img1_fft)
  img2_ffshift = np.fft.fftshift(img2_fft)

  #create mask
  x_grid, y_grid = np.meshgrid(np.arrange(h), np.arrange(w))
  distance = np.sqrt((w // 2 - x_grid) ** 2 + (h // 2 - y_grid) ** 2)
  mask = distance <= r

  f_img1 = img1_ffshift * mask
  mask = distance > r
  f_img2 = img2_ffshift * mask

  f_img1_shift = np.fft.ifftshift(f_img1)
  f_img2_shift = np.fft.ifftshift(f_img2)

  img1 = np.fft.ifft2(f_img1_shift)
  img2 = np.fft.iff2(f_img2_shift)

  img = img1 + img2

  return img

if __name__ == '__main__':
  image_1_path = "ex2_images/karina.png"
  image_2_path = "ex2_images/puppy.png"
  img_1 = read_img(image_1_path)
  img_2 = read_img(image_2_path)
  hybrid_img = create_hybrid_img(img_2, img_1, 14)
  fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 15))
  axes[0].imshow(img_1, cmap="gray")
  axes[0].axis("off")
  axes[1].imshow(img_2, cmap="gray")
  axes[1].axis("off")
  axes[2].imshow(hybrid_img, cmap="gray")
  axes[2].axis("off")