import numpy as np
import pytest

from ripplemapper.classes import RippleContour, RippleImage, RippleImageSeries
from ripplemapper.data.example import (example_contour, example_data,
                                       example_rimgs)
from ripplemapper.io import load


def test_ripple_contour_initialization_from_file():
    contour = RippleContour(example_contour, image=None)
    assert contour.values is not None
    assert contour.method == 'Lower Boundary'

def test_load_ripple_contour():
    contour=load(example_contour)
    assert contour.values is not None
    assert contour.method == 'Lower Boundary'

def test_ripple_contour_initialization_from_values():
    values = np.array([1, 2, 3])
    method = 'test_method'
    contour = RippleContour(values, method)
    assert np.array_equal(contour.values, values)
    assert contour.method == method

def test_ripple_image_initialization_from_file():
    image = RippleImage(example_data[0])
    assert image.image is not None
    assert image.source_file == str(example_data[0].resolve())

def test_load_ripple_image():
    image = load(example_data[0])
    assert image.image is not None
    assert image.source_file == str(example_data[0].resolve())

def test_ripple_image_add_contour():
    image = RippleImage(example_data[0])
    values = np.array([1, 2, 3])
    method = 'test_method'
    image.add_contour(values, method)
    assert len(image.contours) == 1
    assert np.array_equal(image.contours[0].values, values)
    assert image.contours[0].method == method

def test_ripple_image_series_initialization():
    image1 = RippleImage(example_data[0])
    image2 = RippleImage(example_data[1])
    series = RippleImageSeries([image1, image2])
    assert len(series.images) == 2

def test_ripple_image_series_load():
    series = load(example_rimgs)
    assert len(series.images) == 4

def test_ripple_image_save_load(tmp_path):
    image = RippleImage(example_data[0])
    values = np.array([1, 2, 3])
    method = 'test_method'
    image.add_contour(values, method)

    save_path = tmp_path / 'test_image.rimg'
    image.save(fname=str(save_path))

    loaded_image = RippleImage(str(save_path))
    assert len(loaded_image.contours) == 1
    assert np.array_equal(loaded_image.contours[0].values, values)
    assert loaded_image.contours[0].method == method

if __name__ == '__main__':
    pytest.main()
