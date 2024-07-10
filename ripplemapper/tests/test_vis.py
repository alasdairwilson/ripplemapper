from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ripplemapper.visualisation import plot_contours, plot_image


def test_plot_contours(loaded_example_contour):
    plot_contours(loaded_example_contour)
    plt.close()

def test_plot_image(loaded_example_image):
    plot_image(loaded_example_image, include_contours=False)
    plt.close()

def test_plot_image_with_contours(loaded_example_image_with_contours):
    plot_image(loaded_example_image_with_contours)
    plt.close()

def test_plot_contours_calls_plot(loaded_example_contour):
    plt.plot = MagicMock()
    plot_contours(loaded_example_contour)
    assert np.any(plt.plot.called)

def test_plot_image_calls_imshow(loaded_example_image):
    plt.imshow = MagicMock()
    plot_image(loaded_example_image, include_contours=False)
    assert plt.imshow.called
    plt.imshow.assert_called_with(loaded_example_image.image, cmap='gray')

def test_plot_image_with_contours_calls_plot(loaded_example_image_with_contours):
    plt.imshow = MagicMock()
    plt.plot = MagicMock()
    plot_image(loaded_example_image_with_contours)
    assert plt.imshow.called
    assert plt.plot.called
    plt.imshow.assert_called_with(loaded_example_image_with_contours.image, cmap='gray')

if __name__ == '__main__':
    pytest.main()
