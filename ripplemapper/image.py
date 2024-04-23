"""Ripplemapper images module."""
import numpy as np
from skimage import color, feature, filters, morphology
from skimage.segmentation import chan_vese


def preprocess_image(image: np.ndarray, roi_x: list[int]=False, roi_y: list[int]=False, sigma: float=1) -> np.ndarray:
    """
    Preprocess the image by converting it to grayscale and applying Gaussian blur.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    gray_image = color.rgb2gray(image)
    blurred_gray_image = filters.gaussian(gray_image, sigma=sigma)
    # crop the image
    if roi_x:
        blurred_gray_image = blurred_gray_image[roi_x[0]:roi_x[1], :]
    if roi_y:
        blurred_gray_image = blurred_gray_image[:, roi_y[0]:roi_y[1]]
    return blurred_gray_image

def cv_segmentation(image: np.ndarray, use_gradient=True, **kwargs) -> np.ndarray:
    """
    Perform image segmentation using skimage chan-vese method.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Segmented image.
    """
    # Define default values for kwargs in the wrapper
    default_kwargs = {
        'mu': 0.5,
        'dt': 0.5,
        'lambda1': 1,
        'lambda2': 3,
    }

    # Update kwargs with default values if they are not already set
    for key, value in default_kwargs.items():
        kwargs.setdefault(key, value)

    cv = chan_vese(
        image,
        **kwargs
    )
    return cv

def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detection.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Binary edge image.
    """
    edges_gray = feature.canny(image, sigma=1)
    return edges_gray

def process_edges(edges_gray: np.ndarray, sigma: float=0) -> np.ndarray:
    """
    Process the binary edge image by performing morphological operations.

    Parameters:
        edges_gray (numpy.ndarray): Binary edge image.

    Returns:
        numpy.ndarray: Processed edge image.
    """
    edges_dilated = morphology.binary_dilation(edges_gray, footprint=np.ones((5, 5)))
    edges_closed = morphology.binary_closing(edges_dilated, footprint=np.ones((5, 5)))
    edges_cleaned = morphology.remove_small_objects(edges_closed, min_size=300)
    # optionally blur the edges
    if sigma > 0:
        edges_cleaned = filters.gaussian(edges_cleaned, sigma=sigma)
    return edges_cleaned
