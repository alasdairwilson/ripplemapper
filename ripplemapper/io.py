"""This module is for input/output functions."""
import cv2
import numpy as np


#  TODO (ADW): Add support for other image file types just use load_tif for now.
#  should probably be looping in this function rather than the dispatched functions but... it's fine for now.
def load_image(file: str) -> np.ndarray:
    """Load an image file based on file extension."""
    # TODO (ADW): this needs to be refactored to allow lists.
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

def load_dir(directory: str, pattern: str | bool = False, skip: int = 1, start: int=0, end: int=None) -> tuple[list[np.ndarray], list[str]]:
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

    import os

    files = os.listdir(directory)
    files = [file for file in files if file.endswith('.tif') or file.endswith('.tiff')]
    if not files:
        raise FileNotFoundError(f"No tif files found in {directory}")
    if pattern:
        files = [file for file in files if pattern in file]
    files = files[start:end:skip]
    if not files:
        raise FileNotFoundError(f"No tif files found in {directory} matching pattern {pattern}")

    files, img_data = load_tif([os.path.join(directory, file) for file in files])
    return files, img_data

def load_dir_to_obj(directory: str, pattern: str | bool = False, skip: int = False, start: int=0, end: int=-1, **kwargs) -> list:
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
