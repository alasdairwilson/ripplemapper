"""This module is for input/output functions."""
import os
from pathlib import PosixPath, WindowsPath

import cv2
import numpy as np

__all__ = ["load_image", "load_tif", "load_dir", "load_dir_to_obj", "load"]

def load(file: str | PosixPath):
    """Load a file into a ripplemapper object based on file extension."""
    from ripplemapper.classes import (RippleContour, RippleImage,
                                      RippleImageSeries)

    if isinstance(file, PosixPath) | isinstance(file, WindowsPath):
        file = str(file.resolve())
    if file.endswith(".txt") | file.endswith(".json"):
        return RippleContour(file)
    elif file.endswith(".rimg") | file.endswith(".tif") | file.endswith(".tiff"):
        return RippleImage(file)
    elif file.endswith(".rimgs"):
        return RippleImageSeries(file)
    else:
        raise ValueError(f"Unsupported file type: {file}")


#  TODO (ADW): Add support for other image file types just use load_tif for now.
#  should probably be looping in this function rather than the dispatched functions but... it's fine for now.
def load_image(file: str | PosixPath) -> np.ndarray:
    """Load an image file based on file extension."""
    # TODO (ADW): this needs to be refactored to allow lists.
    if isinstance(file, PosixPath) | isinstance(file, WindowsPath):
        file = str(file.resolve())
    if file.endswith('.tif') or file.endswith('.tiff'):
        _, img_data = load_tif(file)
    else:
        raise ValueError(f"Unsupported file type: {file}")
    return img_data[0]


def load_tif(files: str | list[str]) -> list[np.ndarray]:
    """Load an array of tif files and return numpy.ndarray."""

    if isinstance(files, str):
        files = [files]

    img_data = []
    for file in files:
        img = cv2.imread(file)
        img_data.append(img)

    return files, img_data

def load_dir(directory: str | PosixPath, pattern: str | bool = False, skip: int = 1, start: int=0, end: int | bool=None) -> tuple[list[np.ndarray], list[str]]:
    """Load all tif files found in directory and return the data in a list of numpy.ndarray.

    Parameters
    ----------
    directory : str
        directory path to load tif files from
    pattern : str, optional
        optional pattern to match file names, by default False
    skip : int, optional
        number of files to skip, by default False

    Returns
    -------
    tuple[list[np.ndarray], list[str]]
        list of the data arrays extracted from the tif files.
    """
    if isinstance(directory, PosixPath):
        directory = str(directory.resolve())
    files = sorted(os.listdir(directory))
    files = [file for file in files if file.endswith('.tif') or file.endswith('.tiff')]
    if not files:
        raise FileNotFoundError(f"No tif files found in {directory}")
    if pattern:
        files = [file for file in files if pattern in file]
    if end is None:
        end = len(files)

    files = files[start:end:skip]
    if not files:
        raise FileNotFoundError(f"No tif files found in {directory} matching pattern {pattern}")

    files, img_data = load_tif([os.path.join(directory, file) for file in files])
    return files, img_data

def load_dir_to_obj(directory: str | PosixPath, pattern: str | bool = False, skip: int = 1, start: int=0, end: int=None, **kwargs) -> list:
    """Load all tif files found in directory and return the data in a list of Ripple Image objects.

    Parameters
    ----------


    Returns
    -------
    list[RippleImage]
        list of the data arrays extracted from the tif files.
    """
    # prevent circular import
    from ripplemapper.classes import RippleImage
    files, img_data = load_dir(directory, pattern, skip=skip, start=start, end=end)
    return [RippleImage(file, img_data, **kwargs) for file, img_data in zip(files, img_data)]
