import datetime
import os

from cv2 import Mat
from tifffile import tifffile

import cv2
import numpy as np
import tifffile
from PIL import Image



def round_all(t: tuple, precision: int = 1) -> tuple:
    return tuple(round(t[i], precision) for i in range(len(t)))

def load_image_debug(img_path: str, backup_path: str) -> Mat | np.ndarray:
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ImageLoadError(f"cv2.imread failed to load image from initial path: {img_path}")

        backup_image_path = os.path.join(backup_path, *img_path.split(os.sep)[-3:])
        os.makedirs(os.path.dirname(backup_image_path), exist_ok=True)
        cv2.imwrite(backup_image_path, img)
        return img
    except FileNotFoundError:
        print("--- file not found - checking backup")
        relative_path_parts = img_path.split(os.sep)[2:]
        if not relative_path_parts:
            raise ImageLoadError("Relative path parts are empty, cannot construct backup path.")
        backup_image_path = os.path.join(backup_path, *relative_path_parts)
        img = cv2.imread(backup_image_path)
        if img is None:
            raise ImageLoadError(f"cv2.imread failed to load image from backup path: {backup_image_path}")
        os.makedirs(os.path.dirname(backup_image_path), exist_ok=True)
        cv2.imwrite(backup_image_path, img)
        return img

def debug_image(image, title, target_folder, high_quality = False):
    os.makedirs(target_folder, exist_ok=True)
    if high_quality:
        file_name = os.path.join(target_folder, f"{title}.tif")
        save_tiff_cv2(image, file_name)

    file_name = os.path.join(target_folder, f"{title}.jpg")
    cv2.imwrite(file_name, image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])


def save_tiff_cv2(image_data, filepath):
    if image_data.dtype not in [np.uint8, np.uint16]:
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = (image_data * 255).astype(np.uint8)  # Scale and convert
        else:
            raise ValueError("Unsupported image data type.  Must be uint8, uint16, float32, or float64.")

    if image_data.ndim == 3 and image_data.shape[2] == 3:
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        mode = 'RGB'
    elif image_data.ndim == 2:
        mode = 'L'
    else:
        raise ValueError("Unsupported image dimensions.")

    image_pil = Image.fromarray(image_data, mode=mode)
    tifffile.imwrite(filepath, np.array(image_pil), photometric=mode)

def format_time(seconds):
    """Formats a time duration in seconds into DD.HH.MM.SS format."""
    time_delta = datetime.timedelta(seconds=seconds)
    days = time_delta.days
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    days_str = f"{days:02d}"
    hours_str = f"{hours:02d}"
    minutes_str = f"{minutes:02d}"
    seconds_str = f"{seconds:02d}"

    return f"{days_str}.{hours_str}.{minutes_str}.{seconds_str}"