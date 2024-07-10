import numpy as np
import pytest

from ripplemapper.classes import RippleContour, RippleImage, RippleImageSeries
from ripplemapper.data.example import (example_contour, example_data,
                                       example_dir, example_rimgs)
from ripplemapper.io import (load, load_dir, load_dir_to_obj, load_image,
                             load_tif)


def test_load_ripple_contour():
    contour = load(example_contour)
    assert contour.values is not None
    assert contour.method == 'Lower Boundary'
    assert isinstance(contour, RippleContour)

def test_load_ripple_image():
    image = load(example_data[0])
    assert image.image is not None
    assert image.source_file == str(example_data[0].resolve())
    assert isinstance(image, RippleImage)

def test_load_ripple_image_series():
    series = load(example_rimgs)
    assert len(series.images) == 4
    assert isinstance(series, RippleImageSeries)

def test_load_image():
    img_data = load_image(example_data[0])
    assert isinstance(img_data, np.ndarray)
    assert img_data is not None

def test_load_tif():
    files, img_data = load_tif([str(example_data[0])])
    assert len(files) == 1
    assert isinstance(img_data[0], np.ndarray)
    assert img_data[0] is not None

def test_load_dir():
    files, img_data = load_dir(example_dir)
    assert len(files) > 0
    assert isinstance(img_data[0], np.ndarray)
    assert img_data[0] is not None

def test_load_dir_to_obj():
    images = load_dir_to_obj(example_dir)
    assert len(images) > 0
    assert images[0].image is not None
    assert isinstance(images[0], RippleImage)

if __name__ == '__main__':
    pytest.main()
