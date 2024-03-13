"""Module for detecting the air water boundary."""

import numpy as np
from scipy.spatial import cKDTree
from skimage import io, color, filters, feature, morphology, measure, segmentation
from typing import Tuple, List
from ripplemapper.io import load_tif
import matplotlib.pyplot as plt
import cv2
import heapq


def preprocess_image(image: np.ndarray, roi_x: list[int]=False, roi_y: list[int]=False) -> np.ndarray:
    """
    Preprocess the image by converting it to grayscale and applying Gaussian blur.
    
    Parameters:
        image (numpy.ndarray): Input image.
    
    Returns:
        numpy.ndarray: Preprocessed image.
    """
    gray_image = color.rgb2gray(image)
    blurred_gray_image = filters.gaussian(gray_image, sigma=2)
    if roi_x:
        blurred_gray_image = blurred_gray_image[roi_x[0]:roi_x[1], :]
    if roi_y:
        blurred_gray_image = blurred_gray_image[:, roi_y[0]:roi_y[1]]
    return blurred_gray_image

def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detection.
    
    Parameters:
        image (numpy.ndarray): Input image.
    
    Returns:
        numpy.ndarray: Binary edge image.
    """
    edges_gray = feature.canny(image, sigma=1)
    return edges_gray

def process_edges(edges_gray: np.ndarray, sigma: float=0) -> np.ndarray:
    """
    Process the binary edge image by performing morphological operations.
    
    Parameters:
        edges_gray (numpy.ndarray): Binary edge image.
    
    Returns:
        numpy.ndarray: Processed edge image.
    """
    edges_dilated = morphology.binary_dilation(edges_gray, footprint=np.ones((5, 5)))
    edges_closed = morphology.binary_closing(edges_dilated, footprint=np.ones((5, 5)))
    edges_cleaned = morphology.remove_small_objects(edges_closed, min_size=300)
    # optionally blur the edges
    if sigma > 0:
        edges_cleaned = filters.gaussian(edges_cleaned, sigma=sigma)
    return edges_cleaned

def find_contours(edges_cleaned: np.ndarray, level: float=0.5) -> np.ndarray:
    """
    Find contours in the edge image and approximate them to simplify.
    
    Parameters:
        edges_cleaned (numpy.ndarray): Processed edge image.
        tolerance (int): Tolerance value for approximating contours.
    
    Returns:
        numpy.ndarray: Approximated contour vertices.
    """
    contours = measure.find_contours(edges_cleaned, level=level)
    # sort contours by length
    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    return contours


def find_nearest_points(poly_from: np.ndarray, poly_to: np.ndarray) -> np.ndarray:
    """
    For each point in poly_from, find the nearest point in poly_to.
    
    Parameters:
        poly_from (numpy.ndarray): Source vertices.
        poly_to (numpy.ndarray): Target vertices.
        
    Returns:
        numpy.ndarray: Array of nearest points in poly_to for each point in poly_from.
    """
    tree_to = cKDTree(poly_to)
    dist, indices = tree_to.query(poly_from)
    nearest_points = poly_to[indices]
    return nearest_points

def compute_recursive_midpoints(poly_a: np.ndarray, poly_b: np.ndarray, iterations: int) -> np.ndarray:
    """
    Compute midpoints between two contours recursively, addressing the shape mismatch.
    
    Parameters:
        poly_a (numpy.ndarray): Vertices of the first contour.
        poly_b (numpy.ndarray): Vertices of the second contour.
        iterations (int): Number of iterations for recursion.
        
    Returns:
        numpy.ndarray: Midpoint vertices after the final iteration.
    """
    if iterations == 0:
        return poly_a, poly_b  # Or return poly_b, or an average if you prefer.

    # Initialize KD-Trees for each set of points
    tree_a = cKDTree(poly_a)
    tree_b = cKDTree(poly_b)

    # New sets for midpoints
    midpoints_a = np.empty_like(poly_a)
    midpoints_b = np.empty_like(poly_b)

    # Compute midpoints from a to b
    for i, point in enumerate(poly_a):
        dist, index = tree_b.query(point)
        nearest_point = poly_b[index]
        midpoints_a[i] = (point + nearest_point) / 2

    # Compute midpoints from b to a
    for i, point in enumerate(poly_b):
        dist, index = tree_a.query(point)
        nearest_point = poly_a[index]
        midpoints_b[i] = (point + nearest_point) / 2

    # Recursively refine the midpoints
    return compute_recursive_midpoints(midpoints_a, midpoints_b, iterations - 1)

def extend_contour(contour, shape):
    """Extends the contour to the edges of the image region defined by shape.

    Parameters
    ----------
    contour : _type_
        points marking the vertexes of the contour.
    shape : tuple
        len, width of the image.
    """
    # make a new first point and prepend it to the array, the new first point should have x=0 and y= the same as the second point
    print(contour[-5:-1])
    if contour[0][1] > shape[1]/2:
        new_first_point = [contour[0][1], shape[1]]
        new_last_point =  [contour[-1][1], 0]
    else:
        new_first_point = [contour[0][1], 0]
        new_last_point =  [contour[-1][1], shape[0]]

    print(new_last_point, new_first_point)

    contour = np.vstack([new_first_point, contour, new_last_point])



    return contour

def combine_contours(contour1, contour2):
    """Combines two contours into one.

    Parameters
    ----------
    contour1 : _type_
        points marking the vertexes of the first contour.
    contour2 : _type_
        points marking the vertexes of the second contour.
    """
    # we need contour one to run from low to high and contour 2 to run from high to low
    if contour1[0][1] > contour1[0][-1]:
        contour1 = np.flip(contour1)
    if contour2[0][1] < contour2[0][-1]:
        contour2 = np.flip(contour2)
    
    #stitch them together
    contour = np.vstack([contour1, contour2])
    return contour


def distance_map(binary_map):

    # Assuming `binary_map` is your binary image with interiors marked as 1's and exteriors as 0's
    # First, ensure the binary_map is of type uint8
    binary_map = binary_map.astype(np.uint8)

    # Apply the distance transform
    distance_map = cv2.distanceTransform(binary_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) ** 2
    # Optionally, normalize the distance map for visualization
    norm_distance_map = cv2.normalize(distance_map, None, 0, 1.0, cv2.NORM_MINMAX)
    return norm_distance_map


def find_nearest_intersection(contour, point, angles):
    """
    Cast rays from a point at given angles and find the nearest intersection with a contour.
    
    Parameters:
        contour (list of tuple): The contour to intersect.
        point (tuple): The starting point for the rays.
        angles (list of float): Angles at which to cast rays (in radians).
        
    Returns:
        tuple: The closest intersection point on the contour and the midpoint.
    """
    closest_point = None
    shortest_distance = np.inf
    midpoint = None
    
    for angle in angles:
        ray_direction = np.array([np.cos(angle), np.sin(angle)])
        intersection = cast_ray_to_contour(contour, point, ray_direction)
        
        if intersection is not None:
            distance = np.linalg.norm(np.array(point) - np.array(intersection))
            if distance < shortest_distance:
                shortest_distance = distance
                closest_point = intersection
                midpoint = (np.array(point) + np.array(intersection)) / 2
                
    return closest_point, midpoint

def cast_ray_to_contour(contour, start_point, direction):
    """
    Cast a ray from a start point in a given direction and find the intersection with a contour.
    
    This is a placeholder function. Implement the actual ray-contour intersection logic based on your application's specifics.
    
    Parameters:
        contour (list of tuple): The contour to intersect.
        start_point (tuple): The ray's starting point.
        direction (np.array): The ray's direction.
        
    Returns:
        tuple: The intersection point, if any.
    """
    # Placeholder implementation. Replace with actual ray-contour intersection logic.
    return None

def neighbors(node, grid_shape):
    """Generate neighbors for a given node."""
    # 8-connected grid
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in directions:
        if get_next_node(node, dx, dy) is None:
            continue
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
            yield (nx, ny)

def a_star(start, goal, grid):
    """A simple A* algorithm."""
    open_set = []
    heapq.heappush(open_set, (0, start))  # (cost, node)
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
            return path[::-1]  # Return reversed path

        for neighbor in neighbors(current, grid.shape):
            tentative_g_score = g_score[current] + 1 / (grid[neighbor] + 0.01)  # Avoid division by zero
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + np.linalg.norm(np.array(neighbor) - np.array(goal))
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

def get_next_node(node, dx, dy):
    try:
        nx, ny = node[0] + dx, node[1] + dy
    except IndexError:
        return None
    return nx, ny

if __name__ == "__main__":
    # Load and preprocess the image
    image_path = "/mnt/h/1_00369.tif"
    files, img_data = load_tif(image_path)
    gray_image = preprocess_image(img_data, roi_y=[400, 900])

    # Detect edges and process them
    edges_gray = detect_edges(gray_image.squeeze())
    edges_cleaned = process_edges(edges_gray)

    # Find contours and approximate them
    contours = find_contours(edges_cleaned, level=0.5)
    # via a flood fill
    blank_image = np.zeros(edges_cleaned.shape, dtype=np.uint8)
    cont1 = np.flip(contours[0]).astype(np.int32)
    cont2 = np.flip(contours[1]).astype(np.int32)
    contour = combine_contours(cont1, cont2)

    img = cv2.drawContours(blank_image, [contour], 0, (255, 255, 255), -1)
    plt.figure()
    plt.imshow(img)

    distance_map = distance_map(img)
    plt.figure()
    plt.imshow(distance_map)
    
    start = (np.argmax(distance_map[:,0]), 0)  # Highest value on the left boundary
    goal = (np.argmax(distance_map[:, -1]), distance_map.shape[1] - 1)  # Highest value on the right boundary, for simplicity
    path = a_star(start, goal, distance_map)

    # To visualize the path on the distance map for verification
    import matplotlib.pyplot as plt

    plt.imshow(distance_map, cmap='gray')
    plt.plot(start[1], start[0], 'ro')
    plt.plot(goal[1], goal[0], 'go')
    plt.figure()
    print(path[0:5])
    plt.plot([p[1] for p in path], [p[0] for p in path], 'r-')  # x and y are swapped for plotting
    plt.figure()
    plt.plot([distance_map[p[0], p[1]] for p in path], 'r-')

    poly1_vertices = measure.approximate_polygon(contours[0], tolerance=1)
    poly2_vertices = measure.approximate_polygon(contours[1], tolerance=1)

    midpoint_vertices_a, midpoint_vertices_b = compute_recursive_midpoints(poly1_vertices, poly2_vertices, iterations=3)

    # Print or further process the midpoint vertices as needed
    plt.figure()
    plt.imshow(edges_cleaned)
    plt.plot(midpoint_vertices_a[:, 1], midpoint_vertices_a[:, 0], 'r-')
    plt.plot(midpoint_vertices_b[:, 1], midpoint_vertices_b[:, 0], 'r-')

    # plot the original image and all our paths:
    plt.figure(figsize=(24,16))
    plt.imshow(gray_image.squeeze(), cmap='gray')
    plt.plot(cont1[:, 0], cont1[:, 1], 'r-')
    plt.plot(cont2[:, 0], cont2[:, 1], 'b-')
    plt.plot(midpoint_vertices_a[:, 1], midpoint_vertices_a[:, 0], 'g-')
    plt.plot(midpoint_vertices_b[:, 1], midpoint_vertices_b[:, 0], 'y-')
    plt.plot([p[1] for p in path], [p[0] for p in path], 'c-')  # x and y are swapped for plotting
    plt.legend(['Upper Contour', 'Lower Contour', 'Midpoints forward', 'Midpoints Reverse', 'A* Path through volume'])
    plt.show()

    plt.figure()
    plt.plot(cont1[:, 0], cont1[:, 1], 'r-')
    plt.show()

