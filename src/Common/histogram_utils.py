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


def create_histogram_as_image(grayscale_image: np.ndarray,
                              hist_height: int = 300,
                              hist_width: int = 256) -> np.ndarray:
    """
    Creates a histogram graph image from a 2D grayscale image.

    The graph includes:
      - A bar plot of pixel counts for each grayscale intensity (0 to 255)
      - Labels for the x-axis (Pixel Intensity) and y-axis (Number of Pixels)
      - A title for the graph.

    The generated graph is converted to a grayscale (2D) NumPy array.

    Args:
        grayscale_image (np.ndarray): 2D grayscale image (values 0-255).
        hist_height (int): Height (in pixels) of the histogram image.
        hist_width (int): Width (in pixels) of the histogram image.

    Returns:
        np.ndarray: A 2D grayscale image of the histogram.
    """
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
    """
    Compute the pixel intensity corresponding to the given cumulative histogram percentage.

    Args:
        percent (float): A percentage between 0 and 1.
        grayscale_image (Optional[np.ndarray]): Grayscale image to compute histogram if needed.
        hist (Optional[np.ndarray]): Precomputed histogram.
        cumulative_hist (Optional[np.ndarray]): Precomputed cumulative histogram.

    Returns:
        Tuple[int, np.ndarray, np.ndarray]: The pixel intensity, histogram, and cumulative histogram.

    Raises:
        Exception: If percent is not between 0 and 1.
    """
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


def create_histogram_sum_as_image(grayscale_image: np.ndarray,
                                  hist_height: int = 300,
                                  hist_width: int = 256) -> np.ndarray:
    """
    Creates an image showing the cumulative percentage of pixels vs pixel intensity.

    Args:
        grayscale_image (np.ndarray): 2D grayscale image (values 0-255).
        hist_height (int): Height (in pixels) of the resulting image.
        hist_width (int): Width (in pixels) of the resulting image.

    Returns:
        np.ndarray: A 2D grayscale image of the cumulative histogram plot.
    """
    grayscale_image = np.asarray(grayscale_image)
    hist, _ = np.histogram(grayscale_image.flatten(), bins=256, range=(0, 256))
    cumulative_hist = np.cumsum(hist)
    total_pixels = grayscale_image.size
    cumulative_percent = cumulative_hist / total_pixels * 100

    dpi = 100
    fig_width = hist_width / dpi
    fig_height = hist_height / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

    ax.plot(range(256), cumulative_percent, color='green', linewidth=2)
    ax.set_title("Cumulative Percentage vs Pixel Intensity")
    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_yticks(np.arange(0, 101, 5))
    ax.set_xticks(np.arange(0, 256, 25))
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    pil_image = Image.open(buf).convert('L')
    percent_vs_color_image = np.array(pil_image)

    return percent_vs_color_image


def histogram_avg_std(image: np.ndarray) -> Tuple[float, float]:
    """
    Compute the approximate average and standard deviation of pixel intensities in an image.

    Args:
        image (np.ndarray): Input image.

    Returns:
        Tuple[float, float]: Mean and standard deviation.
    """
    counts, bin_edges = np.histogram(image, bins=120)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_count = counts.sum()
    mean_approx = (bin_centers * counts).sum() / total_count
    variance_approx = (counts * (bin_centers - mean_approx) ** 2).sum() / total_count
    std_approx = np.sqrt(variance_approx)
    return mean_approx, std_approx


def round_all(t: tuple, precision: int = 1) -> tuple:
    """
    Rounds all elements in a tuple to the specified precision.

    Args:
        t (tuple): Tuple of numbers.
        precision (int): Number of decimal places to round to.

    Returns:
        tuple: A new tuple with rounded values.
    """
    return tuple(round(t[i], precision) for i in range(len(t)))
