"""
image_processing.py

This module provides functions for image processing operations such as
skeletonization and FFT-based low-pass filtering.
"""
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize, dilation, footprint_rectangle


def get_hist_values(img, mask, arr=[0.8, 0.85, 0.9, 0.95, 0.99, 0.995, 0.997, 0.999]):
    histogram = cv2.calcHist([img], [0], mask, [256], [0, 256]).flatten()
    hist_sum = histogram.sum()
    if hist_sum == 0:
        cumulative_prob = np.cumsum(histogram)
    else:
        cumulative_prob = np.cumsum(histogram) / histogram.sum()

    ret = {}
    for i in range(len(arr)):
        percent = arr[i]
        value = np.searchsorted(cumulative_prob, percent)
        ret[arr[i]] = int(value)
    return ret

def normalize_image(image):
    """
    Normalize the input image for consistent thresholding.
    Converts the image to LAB, applies CLAHE on the L-channel, and returns a BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_normalized = cv2.merge((l_clahe, a, b))
    normalized_image = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    return normalized_image

def gamma_correction(image, gamma):
    """
    Apply gamma correction to the image.
    """
    invGamma = 1.0 / gamma
    table = ([((i / 255.0) ** invGamma) * 255 for i in range(256)])
    table = cv2.UMat(np.array(table, dtype="uint8"))
    return cv2.LUT(image, table.get())


def scale_smooth(img:np.ndarray, scale_factor:int) -> np.ndarray:
    if scale_factor <= 1:
        raise ValueError("scale_factor should be greater than 1.")

    original_height, original_width = img.shape[:2]

    new_width = int(original_width / scale_factor)
    new_height = int(original_height / scale_factor)

    img_small = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    img_smooth = cv2.resize(img_small, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    return img_smooth



def find_contours(mask, min_contour_area, max_contour_area) -> Tuple[np.ndarray, List[any], Tuple[List[any], List[any]]]:

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    if hierarchy is None:
        return np.zeros_like(mask), [], [], [[],[]]

    hierarchy = hierarchy[0]

    new_mask = np.zeros_like(mask)
    new_contours = []
    skipped_large = []
    skipped_small = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if  area < min_contour_area:
            skipped_small.append(cnt)
            continue
        if max_contour_area <= area:
            skipped_large.append(cnt)
            continue
        if hierarchy[idx][3] == -1:
            cv2.drawContours(new_mask, [cnt], -1, 255, cv2.FILLED)
        else:
            cv2.drawContours(new_mask, [cnt], -1, 0, cv2.FILLED)
        new_contours.append(cnt)

    new_contours = tuple(new_contours)
    return new_mask, new_contours, hierarchy, [skipped_small, skipped_large]

def skeletonize_image(binary: np.ndarray, dilation_size:int = 0) -> np.ndarray:
    """
    Skeletonize a binary image.

    Args:
        binary (np.ndarray): Input binary image.

    Returns:
        np.ndarray: Skeletonized image in uint8 format.
    """
    skeleton = skeletonize(binary)
    if dilation_size > 0:
        skeleton = dilation(skeleton, footprint_rectangle((dilation_size, dilation_size)))
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton


def sharpen_image(image):
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def unsharp_masking(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, 0)
    sharpened = np.minimum(sharpened, 255)
    sharpened = sharpened.astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        sharpened = np.where(low_contrast_mask, image, sharpened)
    return sharpened

def multiple_in_mask(img, mask, multiplayer):
    masked_region = img * (mask > 0) * multiplayer
    background_region = img * (mask == 0)
    output_img_float = masked_region + background_region
    output_img_uint8 = np.clip(output_img_float, 0, 255).astype(np.uint8)
    return output_img_uint8

def fft_low_pass(image: np.ndarray, cutoff: int) -> np.ndarray:
    """
    Applies a low-pass filter to an image using FFT.

    Args:
        image (np.ndarray): The input grayscale image.
        cutoff (int): The cutoff frequency. Frequencies higher than this will be attenuated.

    Returns:
        np.ndarray: The filtered image.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with a circle of 1s in the center and 0s elsewhere
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), cutoff, 1, -1)

    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back.astype(np.uint8)

def stretch_colors(image, min_target=0, max_target=255):
    """
    Stretches the color values of an image to cover the full range [min_target, max_target].

    This is done by linearly mapping the original minimum and maximum pixel values
    to the target minimum and maximum values. This can improve contrast and
    make features more visible.

    Args:
        image: The input image (NumPy array). Can be grayscale or color.
        min_target: The desired minimum value in the output image (default: 0).
        max_target: The desired maximum value in the output image (default: 255).

    Returns:
        A NumPy array representing the image with stretched color values.

    Raises:
        TypeError: If the input image is not a NumPy array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")

    # Calculate the original minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    if min_val == max_val:
        # Handle the case where all pixel values are the same
        return np.full_like(image, min_target, dtype=image.dtype)

    # Perform the linear mapping
    scale = (max_target - min_target) / (max_val - min_val)
    stretched_image = (scale * (image - min_val) + min_target).astype(image.dtype)

    return stretched_image

def multiplie(image1, image2):
    image1_float = image1.astype(np.float64) / 255.0  # Normalize to [0, 1]
    image2_float = image2.astype(np.float64) / 255.0  # Normalize to [0, 1]

    multiplied_image = image1_float * image2_float
    return (multiplied_image * 255).astype(np.uint8)  # 0-255

def fft_bandpass_filter(img:np.ndarray, low:float = float('-inf'), high:float = float('inf')):
    """
    Apply a bandpass filter to an image using FFT.

    Parameters:
        img (numpy.ndarray): 2D array representing a grayscale image.
        low (float): Low frequency cutoff (radius in frequency domain).
        high (float): High frequency cutoff (radius in frequency domain).

    Returns:
        numpy.ndarray: The filtered image in the spatial domain.
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols), dtype=np.uint8)

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
    mask[(distance >= low) & (distance <= high)] = 1
    fshift_filtered = fshift * mask

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = img_back.astype(img.dtype)

    return img_back
