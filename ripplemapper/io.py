"""This module is for input/output functions."""

import os
from pathlib import PosixPath, WindowsPath

import cv2
import numpy as np

__all__ = ["load_image", "load_tif", "load_dir", "load_dir_to_obj", "load"]

def load(file: str | PosixPath | WindowsPath, **kwargs):
    """
    Load a file into a ripplemapper object based on file extension.

    Parameters
    ----------
    file : str | PosixPath | WindowsPath
        File path to load.

    Returns
    -------
    RippleContour, RippleImage, or RippleImageSeries
        Loaded ripplemapper object based on file extension.

    Raises
    ------
    ValueError
        If the file type is unsupported.
    """
    from ripplemapper.classes import (RippleContour, RippleImage,
                                      RippleImageSeries)

    if isinstance(file, PosixPath) | isinstance(file, WindowsPath):
        file = str(file.resolve())
    if file.endswith(".txt") | file.endswith(".json"):
        return RippleContour(file, **kwargs)
    elif file.endswith(".rimg") | file.endswith(".tif") | file.endswith(".tiff"):
        return RippleImage(file, **kwargs)
    elif file.endswith(".rimgs"):
        return RippleImageSeries(file, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file}")

def load_image(file: str | PosixPath | WindowsPath) -> np.ndarray:
    """
    Load an image file based on file extension.

    Parameters
    ----------
    file : str | PosixPath | WindowsPath
        File path to load.

    Returns
    -------
    np.ndarray
        Loaded image data.

    Raises
    ------
    ValueError
        If the file type is unsupported.
    """
    if isinstance(file, PosixPath) | isinstance(file, WindowsPath):
        file = str(file.resolve())
    if file.endswith('.tif') or file.endswith('.tiff'):
        _, img_data = load_tif(file)
    else:
        raise ValueError(f"Unsupported file type: {file}")
    return img_data[0]

def load_tif(files: str | list[str]) -> list[np.ndarray]:
    """
    Load an array of tif files and return numpy.ndarray.

    Parameters
    ----------
    files : str | list[str]
        File path or list of file paths to load.

    Returns
    -------
    list[np.ndarray]
        List of loaded image data arrays.
    """
    if isinstance(files, str):
        files = [files]

    img_data = []
    for file in files:
        img = cv2.imread(file)
        img_data.append(img)

    return files, img_data

def load_dir(directory: str | PosixPath, pattern: str | bool = False, skip: int = 1, start: int = 0, end: int | bool = None) -> tuple[list[np.ndarray], list[str]]:
    """
    Load all tif files found in directory and return the data in a list of numpy.ndarray.

    Parameters
    ----------
    directory : str | PosixPath
        Directory path to load tif files from.
    pattern : str | bool, optional
        Optional pattern to match file names, by default False.
    skip : int, optional
        Number of files to skip, by default 1.
    start : int, optional
        Starting index, by default 0.
    end : int | bool, optional
        Ending index, by default None.

    Returns
    -------
    tuple[list[np.ndarray], list[str]]
        List of the data arrays extracted from the tif files and their corresponding file names.

    Raises
    ------
    FileNotFoundError
        If no tif files are found in the directory.
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

def load_dir_to_obj(directory: str | PosixPath, pattern: str | bool = False, skip: int = 1, start: int = 0, end: int = None, **kwargs) -> list:
    """
    Load all tif files found in directory and return the data in a list of RippleImage objects.

    Parameters
    ----------
    directory : str | PosixPath
        Directory path to load tif files from.
    pattern : str | bool, optional
        Optional pattern to match file names, by default False.
    skip : int, optional
        Number of files to skip, by default 1.
    start : int, optional
        Starting index, by default 0.
    end : int, optional
        Ending index, by default None.
    **kwargs
        Additional keyword arguments for the RippleImage initialization.

    Returns
    -------
    list[RippleImage]
        List of RippleImage objects initialized from the tif files.
    """
    from ripplemapper.classes import RippleImage
    files, img_data = load_dir(directory, pattern, skip=skip, start=start, end=end)
    return [RippleImage(file, img_data, **kwargs) for file, img_data in zip(files, img_data)]
