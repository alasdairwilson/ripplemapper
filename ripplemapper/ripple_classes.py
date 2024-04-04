from pathlib import Path

import numpy as np

from ripplemapper.image import preprocess_image
from ripplemapper.io import load_image


class RippleContour:
    """Dataclass for ripple contours."""

    def __init__(self, values: np.ndarray, method: str, image): # we do not type image to prevent crossover typing
        self.values = values
        self.method = method
        self.parent_image = image

    def to_physical(self):
        """Converts the contour to physical units."""
        return

    def write(self, fname: str=False):
        """Write the contour to a file."""
        if not fname:
            fname = f"{self.parent_image.source_file}_{self.method}.txt"
        np.savetxt(fname, self.values)

class RippleImage:
    """Dataclass for ripple images."""


    def __init__(self, *args, roi_x: list[int]=False, roi_y: list[int]=False):
        self.contours: list[RippleContour] = []
        print(args)
        if len(args) == 1:
            if isinstance(args[0], str) or isinstance(args[0], Path):
                self.image = load_image(args[0])
                self.source_file = args[0]
            else:
                raise ValueError("Invalid input, expected a path to an image file or fname, image data pair.")
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], np.ndarray):
                self.source_file = args[0]
                self.image = args[1]
            else:
                raise ValueError("Invalid input, expected a sa path to an image file or fname, image data pair.")
        self.prepped_image = preprocess_image(self.image, roi_x=roi_x, roi_y=roi_y)

    def add_contour(self, values: np.ndarray, method: str):
        """Add a contour to the RippleImage object."""
        self.contours.append(RippleContour(values, method, self))
