"""Ripplemapper images module."""

import numpy as np
from skimage import color, feature, filters, morphology
from skimage.segmentation import chan_vese

__all__ = ["preprocess_image", "cv_segmentation", "detect_edges", "process_edges", "threshold_image"]

def threshold_image(image: np.ndarray, level: float = 0.8) -> np.ndarray:
    """
    Threshold the image to make any pixel above the level equal to the max.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    level : float, optional
        Threshold value, by default 0.8.

    Returns
    -------
    np.ndarray
        Binary image.
    """
    prev_max = np.max(image)
    image = image / prev_max
    image[image > level] = 1
    image *= prev_max
    return image

def preprocess_image(image: np.ndarray, roi_x: list[int] = False, roi_y: list[int] = False, sigma: float = 1) -> np.ndarray:
    """
    Preprocess the image by converting it to grayscale and applying Gaussian blur.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    roi_x : list[int], optional
        Region of interest in the x-dimension, by default False.
    roi_y : list[int], optional
        Region of interest in the y-dimension, by default False.
    sigma : float, optional
        Sigma value for Gaussian blur, by default 1.

    Returns
    -------
    np.ndarray
        Preprocessed image.
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    blurred_gray_image = filters.gaussian(gray_image, sigma=sigma)

    if roi_x:
        blurred_gray_image = blurred_gray_image[roi_x[0]:roi_x[1], :]
    if roi_y:
        blurred_gray_image = blurred_gray_image[:, roi_y[0]:roi_y[1]]

    return blurred_gray_image

def cv_segmentation(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Perform image segmentation using skimage chan-vese method.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    **kwargs
        Additional keyword arguments for chan-vese method.

    Returns
    -------
    np.ndarray
        Segmented image.
    """
    default_kwargs = {
        'mu': 0.2,
        'dt': 0.5,
        'lambda1': 1,
        'lambda2': 1,
        'max_num_iter': 500,
        'tol': 1e-3,
    }

    for key, value in default_kwargs.items():
        kwargs.setdefault(key, value)

    cv = chan_vese(image, **kwargs)
    return cv

def detect_edges(image: np.ndarray, sigma: float = 1, low_threshold: float = 0.1, high_threshold: float = 0.5) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detection.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    sigma : float, optional
        Sigma value for Gaussian filter, by default 1.
    low_threshold : float, optional
        Low threshold for hysteresis, by default 0.1.
    high_threshold : float, optional
        High threshold for hysteresis, by default 0.5.

    Returns
    -------
    np.ndarray
        Binary edge image.
    """
    edges_gray = feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    return edges_gray

def process_edges(edges_gray: np.ndarray, sigma: float = 0) -> np.ndarray:
    """
    Process the binary edge image by performing morphological operations.

    Parameters
    ----------
    edges_gray : np.ndarray
        Binary edge image.
    sigma : float, optional
        Sigma value for Gaussian filter, by default 0.

    Returns
    -------
    np.ndarray
        Processed edge image.
    """
    edges_dilated = morphology.binary_dilation(edges_gray, footprint=np.ones((5, 5)))
    edges_closed = morphology.binary_closing(edges_dilated, footprint=np.ones((5, 5)))
    edges_cleaned = morphology.remove_small_objects(edges_closed, min_size=64)

    if sigma > 0:
        edges_cleaned = filters.gaussian(edges_cleaned, sigma=sigma)

    return edges_closed
