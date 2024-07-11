"""Ripplemapper contours module."""

import heapq

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage import measure

from ripplemapper.image import detect_edges, process_edges

__all__ = ["find_contours", "compute_recursive_midpoints", "extend_contour", "combine_contours", "smooth_contour", "distance_map", "neighbors", "a_star", "get_next_node", "find_boundaries", "find_bump_limits", "smooth_bumps", "average_boundaries"]

def find_contours(edges_cleaned: np.ndarray, level: float = 0.5) -> np.ndarray:
    """
    Find contours in the edge image and approximate them to simplify.

    Parameters
    ----------
    edges_cleaned : np.ndarray
        Processed edge image.
    level : float, optional
        Contour level parameter for the find_contours function, by default 0.5.

    Returns
    -------
    np.ndarray
        Approximated contour vertices.
    """
    contours = measure.find_contours(edges_cleaned, level=level)
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    return contours

def compute_recursive_midpoints(poly_a: np.ndarray, poly_b: np.ndarray, iterations: int) -> np.ndarray:
    """
    Compute midpoints between two contours recursively, addressing the shape mismatch.

    Parameters
    ----------
    poly_a : np.ndarray
        Vertices of the first contour.
    poly_b : np.ndarray
        Vertices of the second contour.
    iterations : int
        Number of iterations for recursion.

    Returns
    -------
    np.ndarray
        Midpoint vertices after the final iteration.
    """
    if iterations == 0:
        return poly_a, poly_b

    tree_a = cKDTree(poly_a)
    tree_b = cKDTree(poly_b)

    midpoints_a = np.empty_like(poly_a)
    midpoints_b = np.empty_like(poly_b)

    for i, point in enumerate(poly_a):
        dist, index = tree_b.query(point)
        nearest_point = poly_b[index]
        midpoints_a[i] = (point + nearest_point) / 2

    for i, point in enumerate(poly_b):
        dist, index = tree_a.query(point)
        nearest_point = poly_a[index]
        midpoints_b[i] = (point + nearest_point) / 2

    return compute_recursive_midpoints(midpoints_a, midpoints_b, iterations - 1)

def extend_contour(contour: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Extend the contour to the edges of the image region defined by shape.

    Parameters
    ----------
    contour : np.ndarray
        Points marking the vertices of the contour.
    shape : tuple
        Length and width of the image.

    Returns
    -------
    np.ndarray
        Extended contour vertices.
    """
    if contour[0][1] > shape[1] / 2:
        new_first_point = [contour[0][1], shape[1]]
        new_last_point = [contour[-1][1], 0]
    else:
        new_first_point = [contour[0][1], 0]
        new_last_point = [contour[-1][1], shape[0]]

    contour = np.vstack([new_first_point, contour, new_last_point])
    return contour

def combine_contours(contour1: np.ndarray, contour2: np.ndarray) -> np.ndarray:
    """
    Combine two contours into one.

    Parameters
    ----------
    contour1 : np.ndarray
        Points marking the vertices of the first contour.
    contour2 : np.ndarray
        Points marking the vertices of the second contour.

    Returns
    -------
    np.ndarray
        Combined contour vertices.
    """
    if contour1[0][1] > contour1[0][-1]:
        contour1 = np.flip(contour1)
    if contour2[0][1] < contour2[0][-1]:
        contour2 = np.flip(contour2)

    contour = np.vstack([contour1, contour2])
    return contour

def smooth_contour(contour: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Smooth a contour by convolving with a small window.

    Parameters
    ----------
    contour : np.ndarray
        Points marking the vertices of the contour.
    window : int, optional
        Size of the smoothing window, by default 3.

    Returns
    -------
    np.ndarray
        Smoothed contour vertices.
    """
    x = np.convolve(contour[:, 0], np.ones(window) / window, mode='valid')
    y = np.convolve(contour[:, 1], np.ones(window) / window, mode='valid')
    return np.vstack([x, y]).T

def distance_map(binary_map: np.ndarray) -> np.ndarray:
    """
    Compute the distance map of a binary image.

    Parameters
    ----------
    binary_map : np.ndarray
        Binary image with interiors marked as 1's and exteriors as 0's.

    Returns
    -------
    np.ndarray
        Normalized distance map.
    """
    binary_map = binary_map.astype(np.uint8)
    distance_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) ** 2
    norm_distance_map = cv2.normalize(distance_map, None, 0, 1.0, cv2.NORM_MINMAX)
    return norm_distance_map

def neighbors(node: tuple, grid_shape: tuple) -> tuple:
    """
    Generate neighbors for a given node.

    Parameters
    ----------
    node : tuple
        The current node coordinates.
    grid_shape : tuple
        Shape of the grid.

    Yields
    ------
    tuple
        Neighboring node coordinates.
    """
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
            yield (nx, ny)

def a_star(start: tuple, goal: tuple, grid: np.ndarray) -> list:
    """
    A simple A* algorithm for pathfinding.

    Parameters
    ----------
    start : tuple
        Starting node coordinates.
    goal : tuple
        Goal node coordinates.
    grid : np.ndarray
        Grid representing the map.

    Returns
    -------
    list
        Path from start to goal.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: np.linalg.norm(np.array(start) - np.array(goal))}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor in neighbors(current, grid.shape):
            tentative_g_score = g_score[current] + 1 / (grid[neighbor] + 0.01)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def get_next_node(node: tuple, dx: int, dy: int) -> tuple:
    """
    Get the next node coordinates by moving in the given direction.

    Parameters
    ----------
    node : tuple
        Current node coordinates.
    dx : int
        Change in x-coordinate.
    dy : int
        Change in y-coordinate.

    Returns
    -------
    tuple
        Next node coordinates, or None if out of bounds.
    """
    try:
        nx, ny = node[0] + dx, node[1] + dy
    except IndexError:
        return None
    return nx, ny

def find_boundaries(gray_image: np.ndarray) -> tuple:
    """
    Find the upper and lower boundaries of the edge region.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image.

    Returns
    -------
    tuple
        Upper and lower boundaries as numpy arrays.
    """
    edges_gray = detect_edges(gray_image)
    edges_cleaned = process_edges(edges_gray)
    contours = find_contours(edges_cleaned, level=0.5)

    if np.mean(contours[0][:, 0]) > np.mean(contours[1][:, 0]):
        upper, lower = contours[0], contours[1]
    else:
        upper, lower = contours[1], contours[0]
    return upper, lower

def find_bump_limits(large_changes: np.array, current: int = 0, max_size: int = 10, bumps: list[tuple[int, int]] = []) -> list[tuple[int, int]]:
    """
    Recursive function to find the limits of "small" bumps in the data.

    Parameters
    ----------
    large_changes : np.array
        Array of indices where large changes in the data occur.
    current : int, optional
        Current index to start from, by default 0.
    max_size : int, optional
        Maximum size of a bump, by default 10.
    bumps : list[tuple[int, int]], optional
        List of tuples representing the start and end indices of each bump, by default [].

    Returns
    -------
    list[tuple[int, int]]
        List of tuples representing the start and end indices of each bump.
    """
    if not (large_changes > current).any():
        return bumps
    start = large_changes[(large_changes > current)][0]
    end = False
    for i in np.arange(1, max_size):
        if start + i > large_changes[-1]:
            return bumps
        if start + i in large_changes:
            end = start + i
    if end:
        bumps.append((start, end))
    current = end or start + max_size
    if current > large_changes[-1]:
        return bumps
    return find_bump_limits(large_changes, current=current, bumps=bumps, max_size=max_size)

def smooth_bumps(contour, max_size: int = 40, std_factor: float = 2.0):
    """
    Function to smooth out bumps in the contour data.

    Parameters
    ----------
    contour : RippleContour
        Contour object containing the data to be smoothed.
    max_size : int, optional
        Maximum size of a bump, by default 40.
    std_factor : float, optional
        Standard deviation factor for identifying large changes, by default 2.0.

    Returns
    -------
    RippleContour
        The smoothed RippleContour object.
    """
    moving_avg = np.convolve(contour.values[0, :], np.ones(100) / 100, mode='valid')
    diffs = contour.values[0, :len(moving_avg)] - moving_avg
    gradients = np.gradient(diffs)
    large_changes = np.where(np.abs(gradients) > std_factor * np.std(gradients))[0]
    if len(large_changes) == 0:
        return contour
    bumps = find_bump_limits(large_changes, max_size=max_size, bumps=[])
    indices = []
    for bump in bumps:
        indices += list(np.arange(bump[0], bump[1]))
    indices = np.array(indices)
    print("num removed", indices.shape)
    contour.values = np.delete(contour.values, indices[indices < contour.values.shape[1]], axis=1)
    return contour

def average_boundaries(self, contour_a = None, contour_b = None, iterations: int = 3, save_both: bool = True) -> np.ndarray:
    """
    Average the two contours to get a more accurate representation of the interface.

    Parameters
    ----------
    contour_a : RippleContour, optional
        The first contour to average, by default self.contours[0].
    contour_b : RippleContour, optional
        The second contour to average, by default self.contours[1].
    iterations : int, optional
        The number of iterations to average the contours over, by default 3.
    save_both : bool, optional
        Whether to save both averaged contours, by default True.

    Returns
    -------
    np.ndarray
        The averaged contour.
    """
    from ripplemapper.classes import RippleContour

    if not contour_a or not contour_b:
        try:
            contour_a = self.contours[0]
            contour_b = self.contours[1]
        except IndexError:
            raise ValueError("No contours found in RippleImage.contours and no explicit contours passed.")

    poly_a = contour_a.values
    poly_b = contour_b.values
    midpoints_a, midpoints_b = compute_recursive_midpoints(poly_a, poly_b, iterations)
    self.contours.append(RippleContour(midpoints_a, "averaged", self))
    if save_both:
        self.contours.append(RippleContour(midpoints_b, "averaged", self))
