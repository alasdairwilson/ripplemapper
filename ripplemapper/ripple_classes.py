import numpy as np

from ripplemapper.image import preprocess_image


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


    def __init__(self, source_file: str, img_data: np.ndarray, roi_x: list[int]=False, roi_y: list[int]=False):
        self.contours: list[RippleContour] = []
        self.source_file = source_file
        self.image = img_data
        self.prepped_image = preprocess_image(self.image, roi_x=roi_x, roi_y=roi_y)

    def add_contour(self, values: np.ndarray, method: str):
        """Add a contour to the RippleImage object."""
        self.contours.append(RippleContour(values, method, self))
