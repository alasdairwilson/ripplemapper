import numpy as np
import pytest
from skimage import data

from ripplemapper.image import (cv_segmentation, detect_edges,
                                preprocess_image, process_edges,
                                threshold_image)


def test_preprocess_image():
    image = np.array([data.camera(), data.camera(), data.camera()]).T
    preprocessed_image = preprocess_image(image, roi_x=[100, 200], roi_y=[100, 200], sigma=1)
    assert preprocessed_image.shape == (100, 100)  # Verify the ROI dimensions
    assert preprocessed_image.dtype == np.float64  # Verify the image is in float format

def test_cv_segmentation():
    image = data.camera()
    segmented_image = cv_segmentation(image, mu=0.2, dt=0.5, lambda1=1, lambda2=1, max_num_iter=100, tol=1e-3)
    assert segmented_image.shape == image.shape  # Verify the output shape matches input shape
    assert segmented_image.dtype == bool  # Verify the output is a boolean array

def test_detect_edges():
    image = data.camera()
    edges = detect_edges(image, sigma=1, low_threshold=0.1, high_threshold=0.5)
    assert edges.shape == image.shape  # Verify the output shape matches input shape
    assert edges.dtype == bool  # Verify the output is a boolean array

def test_process_edges():
    image = data.camera()
    edges = detect_edges(image, sigma=1)
    processed_edges = process_edges(edges, sigma=0)
    assert processed_edges.shape == edges.shape  # Verify the output shape matches input shape
    assert processed_edges.dtype == bool  # Verify the output is a boolean array

def test_threshold_image():
    image = data.camera().astype(np.float64) / 255  # Normalize image to [0, 1]
    thresholded_image = threshold_image(image, level=0.8)
    assert thresholded_image.shape == image.shape  # Verify the output shape matches input shape
    assert thresholded_image.dtype == image.dtype  # Verify the output dtype matches input dtype
    assert np.max(thresholded_image) == 1.0  # Verify the max value is correct
    assert np.min(thresholded_image) >= 0.0  # Verify the min value is correct

if __name__ == '__main__':
    pytest.main()
