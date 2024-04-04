"""Mostly a collection of functions to help instantiate a list of image classes and add some contours"""
import numpy as np

from ripplemapper.contour import a_star, distance_map, find_contours
from ripplemapper.image import detect_edges, process_edges
from ripplemapper.ripple_classes import RippleImage


def add_boundary_contours(ripple_images: list[RippleImage] or RippleImage) -> list[RippleImage]:
    """Add boundary contours to a list of RippleImage objects."""
    if isinstance(ripple_images, RippleImage):
        ripple_images = [ripple_images]
    for ripple_image in ripple_images:
        edges = detect_edges(ripple_image.prepped_image)
        processed_edges = process_edges(edges)
        contours = find_contours(processed_edges)
        ripple_image.add_contour(contours[0], 'upper')
        ripple_image.add_contour(contours[1], 'lower')

def add_a_star_contours(ripple_images: list[RippleImage] or RippleImage) -> list[RippleImage]:
    """Add A* contours to a list of RippleImage objects."""
    if isinstance(ripple_images, RippleImage):
        ripple_images = [ripple_images]
    for ripple_image in ripple_images:
        d_map = distance_map(ripple_image.prepped_image)
        start = (np.argmax(d_map[:,0]), 0)  # Pixel with the highest value on the left.
        goal = (np.argmax(d_map[:, -1]), d_map.shape[1] - 1)  # Highest value on the right.
        path = a_star(start, goal, d_map)
        ripple_image.add_contour(path, 'a_star_traversal')
