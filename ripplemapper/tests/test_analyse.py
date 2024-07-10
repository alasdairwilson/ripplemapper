import pytest

from ripplemapper.analyse import (add_a_star_contours, add_boundary_contours,
                                  add_chan_vese_contours, remove_small_bumps,
                                  remove_small_bumps_from_images)


def test_add_boundary_contours(loaded_example_image):
    add_boundary_contours(loaded_example_image)
    assert len(loaded_example_image.contours) == 2
    assert loaded_example_image.contours[0].method == 'Upper Boundary'
    assert loaded_example_image.contours[1].method == 'Lower Boundary'

def test_add_a_star_contours(loaded_example_image):
    add_boundary_contours(loaded_example_image)
    add_a_star_contours(loaded_example_image)
    assert len(loaded_example_image.contours) == 3
    assert loaded_example_image.contours[2].method == 'A* traversal'

def test_add_chan_vese_contours(loaded_example_image):
    add_chan_vese_contours(loaded_example_image)
    assert len(loaded_example_image.contours) == 1
    assert loaded_example_image.contours[0].method == 'Chan-Vese'

def test_remove_small_bumps(loaded_example_contour):
    smoothed_contour = remove_small_bumps(loaded_example_contour)
    assert smoothed_contour.values.shape[1] <= loaded_example_contour.values.shape[1]

def test_remove_small_bumps_from_images(loaded_example_image):
    add_boundary_contours(loaded_example_image)
    remove_small_bumps_from_images(loaded_example_image)
    assert len(loaded_example_image.contours) == 2

def test_add_boundary_contours_image_series(loaded_example_image_series):
    add_boundary_contours(loaded_example_image_series)
    for image in loaded_example_image_series.images:
        assert len(image.contours) == 2
        assert image.contours[0].method == 'Upper Boundary'
        assert image.contours[1].method == 'Lower Boundary'

def test_add_a_star_contours_image_series(loaded_example_image_series):
    add_boundary_contours(loaded_example_image_series)
    add_a_star_contours(loaded_example_image_series)
    for image in loaded_example_image_series.images:
        assert len(image.contours) == 3
        assert image.contours[2].method == 'A* traversal'

def test_add_chan_vese_contours_image_series(loaded_example_image_series):
    add_chan_vese_contours(loaded_example_image_series)
    for image in loaded_example_image_series.images:
        assert len(image.contours) == 1
        assert image.contours[0].method == 'Chan-Vese'

def test_remove_small_bumps_from_image_series(loaded_example_image_series):
    add_boundary_contours(loaded_example_image_series)
    remove_small_bumps_from_images(loaded_example_image_series)
    for image in loaded_example_image_series.images:
        assert len(image.contours) == 2

if __name__ == '__main__':
    pytest.main()
