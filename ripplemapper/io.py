"""This module is for input/output functions."""
import cv2
import numpy as np


def load_tif(files: str | list[str]) -> list[np.ndarray]:
    """Load an array of tif files and return numpy.ndarray."""

    if isinstance(files, str):
        files = [files]

    img_data = []
    for file in files:
        img = cv2.imread(file)
        img_data.append(img)

    return files, img_data

def load_directory(directory: str, pattern: str | bool = False) -> tuple[list[np.ndarray], list[str]]:
    """Load all tif files found in directory and return the data in a list of numpy.ndarray.

    Parameters
    ----------
    directory : str
        directory path to load tif files from
    pattern : str, optional
        optional pattern to match file names, by default False

    Returns
    -------
    tuple[list[np.ndarray], list[str]]
        list of the data arrays extracted from the tif files.
    """

    import os

    files = os.listdir(directory)
    files = [file for file in files if file.endswith('.tif') or file.endswith('.tiff')]
    if not files:
        raise FileNotFoundError(f"No tif files found in {directory}")
    if pattern:
        files = [file for file in files if pattern in file]
    if not files:
        raise FileNotFoundError(f"No tif files found in {directory} matching pattern {pattern}")

    files, img_data = load_tif([os.path.join(directory, file) for file in files])
    return files, img_data
