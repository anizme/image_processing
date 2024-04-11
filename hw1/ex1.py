import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    return cv.imread('images/uet.png')

# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()


# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    height, width = image.shape[:2]
    img_gray = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            R, G, B = image[i, j]
            p = 0.299 * R + 0.587 * G + 0.114 * B
            img_gray[i, j] = p
    
    return img_gray
    # grayscale_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # return grayscale_image


# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv.imwrite(output_path, image)


# flip an image as function 
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    flipped_img = cv.flip(image, 1)
    return flipped_img


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


if __name__ == "__main__":
    # Load an image from file
    img = load_image("images/uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_flipped, "images/gray_flipped.jpg")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/gray_rotated.jpg")

    # Show the images
    plt.show() 
