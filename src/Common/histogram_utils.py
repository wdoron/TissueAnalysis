"""
histogram_utils.py

This module provides functions for creating and analyzing histograms of images.
It includes functionality for generating histogram images, computing cumulative
histograms, and calculating statistical measures.
"""

import io
from typing import Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

def create_histogram_as_image(grayscale_image: np.ndarray,
                              hist_height: int = 300,
                              hist_width: int = 256) -> np.ndarray:
    grayscale_image = np.asarray(grayscale_image)
    hist, bins = np.histogram(grayscale_image.flatten(), bins=256, range=(0, 256))

    dpi = 100
    fig_width = hist_width / dpi
    fig_height = hist_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.bar(range(256), hist, color='gray', edgecolor='black')
    ax.set_title("Histogram: Number of Pixels vs Colors")
    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Number of Pixels")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    pil_image = Image.open(buf).convert('L')
    histogram_image = np.array(pil_image)

    return histogram_image


def get_histogram_percentage(percent: float,
                             grayscale_image: Optional[np.ndarray] = None,
                             hist: Optional[np.ndarray] = None,
                             cumulative_hist: Optional[np.ndarray] = None,
                             mask = None) -> Tuple[int, np.ndarray, np.ndarray]:
    if percent > 1 or percent < 0:
        raise Exception(f"Invalid percent range {percent}")

    if cumulative_hist is None:
        if hist is None:
            hist = cv2.calcHist([grayscale_image], [0], mask, [256], [0, 256]).flatten()
        hist_sum = hist.sum()
        if hist_sum == 0 :
            cumulative_hist = np.cumsum(hist)
        else:
            cumulative_hist = np.cumsum(hist) / hist.sum()

    intensity = int(np.searchsorted(cumulative_hist, percent))
    return intensity, hist, cumulative_hist

def histogram_avg_std(image: np.ndarray) -> Tuple[float, float]:
    counts, bin_edges = np.histogram(image, bins=120)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_count = counts.sum()
    mean_approx = (bin_centers * counts).sum() / total_count
    variance_approx = (counts * (bin_centers - mean_approx) ** 2).sum() / total_count
    std_approx = np.sqrt(variance_approx)
    return mean_approx, std_approx
