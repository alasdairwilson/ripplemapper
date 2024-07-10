import numpy as np
import pytest
from skimage import data

from ripplemapper.classes import RippleContour
from ripplemapper.contour import (a_star, combine_contours,
                                  compute_recursive_midpoints, distance_map,
                                  extend_contour, find_boundaries,
                                  find_bump_limits, find_contours,
                                  smooth_bumps, smooth_contour)
from ripplemapper.data.example import example_contour


def test_find_contours():
    image = data.camera()
    edges = np.zeros_like(image)
    edges[100:200, 100:200] = 1
    contours = find_contours(edges, level=0.5)
    assert len(contours) > 0
    assert all(len(c) > 0 for c in contours)

def test_compute_recursive_midpoints():
    poly_a = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    poly_b = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
    midpoints_a, midpoints_b = compute_recursive_midpoints(poly_a, poly_b, iterations=3)
    assert midpoints_a.shape == poly_a.shape
    assert midpoints_b.shape == poly_b.shape

def test_extend_contour():
    contour = np.array([[0, 0], [1, 1], [2, 2]])
    extended_contour = extend_contour(contour, (3, 3))
    assert len(extended_contour) == len(contour) + 2

def test_combine_contours():
    contour1 = np.array([[0, 0], [1, 1], [2, 2]])
    contour2 = np.array([[2, 2], [3, 3], [4, 4]])
    combined_contour = combine_contours(contour1, contour2)
    assert len(combined_contour) == len(contour1) + len(contour2)

def test_smooth_contour():
    contour = np.array([[0, 0], [1, 1], [2, 2]])
    smoothed_contour = smooth_contour(contour, window=2)
    assert len(smoothed_contour) == len(contour) - 1

def test_distance_map():
    binary_map = np.zeros((5, 5), dtype=np.uint8)
    binary_map[2, 2] = 1
    dist_map = distance_map(binary_map)
    assert dist_map.shape == binary_map.shape

def test_a_star():
    grid = np.ones((5, 5))
    start = (0, 0)
    goal = (4, 4)
    path = a_star(start, goal, grid)
    assert len(path) > 0

def test_find_boundaries():
    image = data.camera()
    upper, lower = find_boundaries(image)
    assert upper is not None
    assert lower is not None

def test_find_bump_limits():
    large_changes = np.array([0, 2, 5, 10, 12, 15])
    bumps = find_bump_limits(large_changes, max_size=3)
    assert len(bumps) > 0

def test_smooth_bumps():
    contour = RippleContour(example_contour)
    vals = len(contour.values[0])
    smoothed_contour = smooth_bumps(contour)
    assert smoothed_contour is not None
    assert len(smoothed_contour.values[0]) < vals

if __name__ == '__main__':
    pytest.main()
