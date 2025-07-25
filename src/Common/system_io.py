"""
system_io.py

This module provides functions for image input and output operations,
including loading images, displaying images, and showing contours.
"""
import json
import os

import cv2
import math
from typing import List, Union, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


def _convert_numpy_types(obj):
    """
    Recursively convert NumPy data types in a data structure to native Python types.
    """
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(element) for element in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    return img

def get_csv_filepaths_by_id_folder(root_folder: str) -> Dict[str, List[str]]:
    id_folder_to_csv_filepaths: Dict[str, List[str]] = {}

    for id_folder_name in os.listdir(root_folder):
        id_folder_path = os.path.join(root_folder, id_folder_name)
        if not os.path.isdir(id_folder_path):
            continue

        csv_filepaths: List[str] = []
        for tile_folder_name in os.listdir(id_folder_path):
            tile_folder_path = os.path.join(id_folder_path, tile_folder_name)
            if not os.path.isdir(tile_folder_path):
                continue

            for filename in os.listdir(tile_folder_path):
                if filename.lower().endswith(".csv"):
                    csv_filepath = os.path.join(tile_folder_path, filename)
                    if os.path.isfile(csv_filepath):
                        csv_filepaths.append(csv_filepath)
        if csv_filepaths:
            id_folder_to_csv_filepaths[id_folder_name] = csv_filepaths

    return id_folder_to_csv_filepaths

def save_csv_data_frame(all_data: List[DataFrame], merged_csv_path:str):
    if os.path.exists(merged_csv_path):
        os.remove(merged_csv_path)

    merged_df = pd.concat(all_data, ignore_index=True)
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged CSV data saved to: {merged_csv_path}")

def show_image(image: np.ndarray, title: str = "Image") -> None:
    """
    Display an image using OpenCV's GUI.

    Args:
        image (np.ndarray): The image to display.
        title (str): Window title.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_images(images_input: Union[List[np.ndarray], np.ndarray, Tuple[np.ndarray, ...]], title: str = "image") -> None:
# def display_images(images_input: Union[List[np.ndarray], np.ndarray], title: str = "image") -> None:
    """
    Display one or more images using matplotlib.

    Args:
        images_input (Union[List[np.ndarray], np.ndarray]): A single image or a list of images.
        title (str): The title for the display window.
    """

    def _prepare_image_to_show(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Input must be a NumPy array. Got {type(image)} instead.")
        if image.dtype == bool:
            image = (image * 255).astype(np.uint8)
        elif image.ndim == 2:
            if image.dtype in [np.int32, np.int64]:
                # Assume it's a marker image (integer label map)
                markers_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                rgb_image = cv2.applyColorMap(markers_normalized, cv2.COLORMAP_JET)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    if isinstance(images_input, np.ndarray):
        images = [images_input] if hasattr(images_input, 'ndim') else []
    else:
        images = [img for img in images_input if hasattr(img, 'ndim')]

    n = len(images)
    rows = 1 if n == 1 else 2
    cols = math.ceil(n / rows)

    plt.figure(figsize=(15, 6))
    plt.suptitle(title, fontsize=16)

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 1:
            # Plot as histogram (bar plot)
            plt.bar(range(len(img)), img, color='blue')
            plt.title(f'Histogram {i + 1}')
            plt.grid(True)
        else:
            img_prepared = _prepare_image_to_show(img)
            plt.imshow(img_prepared)
            plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def create_contours(img: np.ndarray, contours: List[np.ndarray], thickness: int = cv2.FILLED) -> None:
    """
    Draw contours on a copy of the image and display it.

    Args:
        img (np.ndarray): The source image.
        contours (List[np.ndarray]): List of contours to draw.
        name (str): Title for the displayed image.
        thickness (int): Thickness of the drawn contours (default uses cv2.FILLED).
    """
    region = img.copy()
    return cv2.drawContours(region, contours, -1, 255, thickness=thickness)

def show_contours(img: np.ndarray, contours: List[np.ndarray], name: str, thickness: int = cv2.FILLED) -> None:
    """
    Draw contours on a copy of the image and display it.

    Args:
        img (np.ndarray): The source image.
        contours (List[np.ndarray]): List of contours to draw.
        name (str): Title for the displayed image.
        thickness (int): Thickness of the drawn contours (default uses cv2.FILLED).
    """
    region = img.copy()
    region = cv2.drawContours(region, contours, -1, 255, thickness=thickness)
    display_images(region, name)

def save_data_as_json(data, path):
    """
    Saves Python data to a JSON file.

    Args:
        data: The Python data structure (dict, list, etc.) to be saved as JSON.
        path: The file path where the JSON file will be saved.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=lambda o: str(o))

def save_csv(results_table: Dict[int, Dict[str, Any]] | None,
             file_name: str,
             skip:List = [], complex_keys = []) -> None:
    """
    Saves the results_table dictionary to a CSV file in tabular format.

    Parameters:
    - results_table (Dict[int, Dict[str, Any]]):
        A dictionary where each key is an index (e.g., int) and the value is another dictionary
        containing various metrics and data.
    - file_name (str):
        The path to the CSV file where the data will be saved.

    Returns:
    - None
    """
    if results_table is None or len(results_table) == 0:
        pd.DataFrame([]).to_csv(file_name, index=False, encoding='utf-8')
        return
    # Convert the results_table dictionary to a list of dictionaries
    data_list = []
    for idx, data in results_table.items():
        serialized_data = data.copy()
        for key in complex_keys:
            if key in serialized_data and key not in skip:
                converted_data = _convert_numpy_types(serialized_data[key])
                serialized_data[key] = json.dumps(converted_data)

        serialized_data['idx'] = idx

        data_list.append(serialized_data)
    df = pd.DataFrame(data_list)
    cols = ['idx'] + [col for col in df.columns if col != 'idx' and col not in skip]
    df = df[cols]

    try:
        df.to_csv(file_name, index=False, encoding='utf-8')

    except Exception as e:
        print(f"An error occurred while saving to CSV: {e}")
        raise e