import pytest

from ripplemapper.data.example import (example_contour, example_data,
                                       example_rimgs)


@pytest.fixture
def loaded_example_contour():
    from ripplemapper.io import load
    return load(example_contour)

@pytest.fixture
def loaded_example_image():
    from ripplemapper.io import load
    return load(example_data[0])

@pytest.fixture
def loaded_example_image_with_contours(loaded_example_contour):
    from ripplemapper.io import load
    image = load(example_data[0])
    image.add_contour(loaded_example_contour.values, loaded_example_contour.method)
    return image

@pytest.fixture
def loaded_example_image_series():
    from ripplemapper.io import load
    return load(example_rimgs)
