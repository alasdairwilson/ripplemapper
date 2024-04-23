from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ripplemapper.contour import smooth_bumps
from ripplemapper.image import preprocess_image
from ripplemapper.io import load_image
from ripplemapper.visualisation import plot_contours, plot_image


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

    def plot(self, *args, **kwargs):
        """Plot the image with contours."""
        plot_contours(self, *args, **kwargs)
        plt.title(f"{self.parent_image.source_file} - Contour: {self.method}")
        return

    def smooth(self, **kwargs):
        """Smooth the contour."""
        smooth_bumps(self, **kwargs)
        return


class RippleImage:
    """Class for ripple images."""

    def __init__(self, *args, roi_x: list[int]=False, roi_y: list[int]=False):
        self.contours: list[RippleContour] = []
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
        self.image = preprocess_image(self.image, roi_x=roi_x, roi_y=roi_y)

    def __repr__(self) -> str:
        return f"RippleImage: {self.source_file.split('/')[-1]}"

    def add_contour(self, values: np.ndarray, method: str):
        """Add a contour to the RippleImage object."""
        self.contours.append(RippleContour(values, method, self))

    def smooth_contours(self, **kwargs):
        """Smooth all the contours in the image."""
        self.contours = [contour.smooth(**kwargs) for contour in self.contours]
        return

    def plot(self, include_contours: bool=True, *args, **kwargs):
        """Plot the image with optional."""
        plot_image(self, include_contours=include_contours, *args, **kwargs)
        plt.title(self.source_file.split('/')[-1])
        return

class RippleImageSeries:
    """Class for a series of ripple images."""

    def __init__(self, image_list: list[RippleImage]):
        self.images = image_list

    def __repr__(self) -> str:
        return f"RippleImageSeries: {len(self.images)} images"

    def animate(self, fname: str=False, **kwargs):
        """Animate the images."""
        if not fname:
            fname = f"{self.images[0].source_file.split('/')[-1]}_animation.gif"
        fig = plt.figure()
        ani = FuncAnimation(fig, self.update, fargs=kwargs, frames=range(len(self.images)), interval=200, repeat=False)
        ani.save(fname, writer='ffmpeg')

    def update(self, frame, **kwargs):
        plt.clf()
        self.images[frame].plot(**kwargs)
