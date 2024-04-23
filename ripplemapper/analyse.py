"""Mostly a collection of functions to help instantiate a list of image classes and add some contours"""
import warnings

import cv2
import numpy as np

from ripplemapper.classes import RippleContour, RippleImage
from ripplemapper.contour import (a_star, combine_contours, distance_map,
                                  find_contours, smooth_bumps)
from ripplemapper.image import cv_segmentation, detect_edges, process_edges


def add_boundary_contours(ripple_images: list[RippleImage] or RippleImage) -> list[RippleImage]:
    """Add boundary contours to a list of RippleImage objects."""
    if isinstance(ripple_images, RippleImage):
        ripple_images = [ripple_images]
    for ripple_image in ripple_images:
        edges = detect_edges(ripple_image.image)
        processed_edges = process_edges(edges)
        contours = find_contours(processed_edges)
        ripple_image.add_contour(np.array([contours[0][:,0],contours[0][:,1]]), 'Upper Boundary')
        ripple_image.add_contour(np.array([contours[1][:,0],contours[1][:,1]]), 'Lower Boundary')


def add_a_star_contours(ripple_images: list[RippleImage] | RippleImage, contour_index: list[int]=[0,1]) -> list[RippleImage]:
    """Add A* contours to a list of RippleImage objects."""
    if len(contour_index) != 2:
        raise ValueError("contour_index must be a list of two integers.")
    if isinstance(ripple_images, RippleImage):
        ripple_images = [ripple_images]
    for ripple_image in ripple_images:
        if len(ripple_image.contours) < 2:
            warnings.warn(f"RippleImage object must have at least two contours, skipping image: {ripple_image.source_file}")
        cont1 = np.flip(ripple_image.contours[contour_index[0]].values).astype(np.int32).T
        cont2 = np.flip(ripple_image.contours[contour_index[1]].values).astype(np.int32).T
        contour = combine_contours(cont1, cont2)
        bounded_img = np.zeros(ripple_image.image.shape, dtype=np.uint8)
        bounded_img = cv2.drawContours(bounded_img, [contour], 0, (255, 255, 255), -1)
        d_map = distance_map(bounded_img)
        start = (np.argmax(d_map[:,0]), 0)  # Pixel with the highest value on the left.
        goal = (np.argmax(d_map[:, -1]), d_map.shape[1] - 1)  # Highest value on the right.
        path = a_star(start, goal, d_map)
        # return type from my a_star function is a list of tuples, need to convert it to a numpy array
        path = np.flip(np.array(path), axis=0).T # the path output has insane shape, need to flip it
        ripple_image.add_contour(path, 'A* traversal')

def add_chan_vese_contours(ripple_images: list[RippleImage] | RippleImage, **kwargs):
    """Add Chan-Vese contours to a list of RippleImage objects."""
    if isinstance(ripple_images, RippleImage):
        ripple_images = [ripple_images]
    for ripple_image in ripple_images:
        cv = cv_segmentation(ripple_image.image, **kwargs)
        contours = find_contours(cv)
        ripple_image.add_contour(np.array([contours[0][:,0],contours[0][:,1]]), 'Chan-Vese')

def remove_small_bumps(contour: RippleContour, **kwargs) -> RippleContour:
    """Remove small bumps from a RippleContour object."""
    return smooth_bumps(contour, **kwargs)

def remove_small_bumps_from_images(ripple_images: list[RippleImage] | RippleImage, **kwargs) -> list[RippleImage]:
    """Remove small bumps from a list of RippleImage objects."""
    if isinstance(ripple_images, RippleImage):
        ripple_images = [ripple_images]
    for ripple_image in ripple_images:
        for contour in ripple_image.contours:
            if contour is not None:
                remove_small_bumps(contour, **kwargs)
    return ripple_images
