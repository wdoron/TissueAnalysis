from typing import List, Tuple
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import shannon_entropy
from skimage.morphology import skeletonize, dilation, footprint_rectangle


def sample_widths_in_mask_distance(seed_num, image_mask: np.ndarray, samples: int) -> float:
    mask = image_mask.astype(bool)
    dt = distance_transform_edt(mask)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0

    rng = np.random.RandomState(seed_num)  # Create consistent random with the ID as seed
    indices = rng.choice(len(ys), size=samples, replace=True)
    sampled_dt = dt[ys[indices], xs[indices]]
    return 2 * np.mean(sampled_dt)



def normalize_image(image, tileGridSize, clipLimit):

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    if len(image.shape) == 2:  # Grayscale
        return clahe.apply(image)

    elif len(image.shape) == 3:
        channels = image.shape[2]
        if channels == 3:  # BGR
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            lab_normalized = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
        else:
            # Multichannel CLAHE per channel
            return cv2.merge([clahe.apply(image[:, :, i]) for i in range(channels)])

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def create_circles_mask(mask: np.ndarray, radius: int = 500) -> np.ndarray:
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    circles_mask = np.zeros_like(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(circles_mask, (cx, cy), radius, 255, thickness=-1)
    return circles_mask

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
    skeleton = skeletonize(binary)
    if dilation_size > 0:
        skeleton = dilation(skeleton, footprint_rectangle((dilation_size, dilation_size)))
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton

def multiple_in_mask(img, mask, multiplayer):
    masked_region = img * (mask > 0) * multiplayer
    background_region = img * (mask == 0)
    output_img_float = masked_region + background_region
    output_img_uint8 = np.clip(output_img_float, 0, 255).astype(np.uint8)
    return output_img_uint8

def fft_bandpass_filter(img:np.ndarray, low:float = float('-inf'), high:float = float('inf')):
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