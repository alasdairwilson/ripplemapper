"""Ripplemapper images module."""
import numpy as np
from skimage import color, feature, filters, morphology
from skimage.segmentation import chan_vese

__all__ = ["preprocess_image", "cv_segmentation", "detect_edges", "process_edges", "threshold_image"]

def threshold_image(image: np.ndarray, level=0.8) -> np.ndarray:
    """
    Threshold the image to make any pixel above the level equal to the max.

    Parameters:
        image numpy.ndarray: Input image.
        level float: Threshold value.

    Returns:
        numpy.ndarray: Binary image.
    """
    image = np.where(image > level, np.max(image))
    return image


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

def cv_segmentation(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Perform image segmentation using skimage chan-vese method.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Segmented image.
    """
    # Define default values for kwargs in the wrapper
    default_kwargs = {
        'mu': 0.2,
        'dt': 0.5,
        'lambda1': 1,
        'lambda2': 1,
        'max_num_iter': 500,
        'tol': 1e-3,
    }

    # Update kwargs with default values if they are not already set
    for key, value in default_kwargs.items():
        kwargs.setdefault(key, value)

    cv = chan_vese(
        image,
        **kwargs
    )
    return cv

def detect_edges(image: np.ndarray, sigma=1, low_threshold: np.float32=0.1, high_threshold: np.float32=0.5) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detection.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: Binary edge image.
    """
    edges_gray = feature.canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
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
    edges_cleaned = morphology.remove_small_objects(edges_closed, min_size=64)
    # optionally blur the edges
    if sigma > 0:
        edges_cleaned = filters.gaussian(edges_cleaned, sigma=sigma)
    return edges_closed


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from ripplemapper.analyse import (add_a_star_contours,
                                      add_boundary_contours,
                                      add_chan_vese_contours)
    from ripplemapper.io import load_dir_to_obj

    x = load_dir_to_obj('/mnt/h/Downloads/D16H10/high_speed/', roi_x=[400,900], skip=25, start=200, end=250)
    # calculate divergence of image
    # img = cv2.GaussianBlur(np.sum(np.abs(np.gradient(x[1].image)), axis=0)/np.max(x[1].image), (11,11), 0)*cv2.GaussianBlur(1-(x[1].image/np.max(x[1].image)), (11,11), 1)
    # img = img/np.max(img)
    # x[1].image = img
    add_boundary_contours(x[1], sigma=2)
    add_a_star_contours(x[1])
    add_chan_vese_contours(x[1], use_gradients=True)
    x[1].plot()


    plt.show()
