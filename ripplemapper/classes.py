import gzip
import json
import pickle
from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ripplemapper.contour import smooth_bumps
from ripplemapper.image import preprocess_image
from ripplemapper.io import load_image
from ripplemapper.visualisation import plot_contours, plot_image

__all__ = ['RippleContour', 'RippleImage', 'RippleImageSeries']

class RippleContour:
    """Dataclass for ripple contours."""

    def __init__(self, *args, image=None):  # we do not type image to prevent crossover typing
        if len(args) == 1 and isinstance(args[0], str) and str(args[0]).endswith('.txt'):
            self._load(args[0], image)
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], str):
            self.values = args[0]
            self.method = args[1]
            self.parent_image = image
        else:
            raise ValueError("Invalid input, expected a file path to a contour file or a values, method pair.")

    def to_physical(self):
        """Converts the contour to physical units."""
        return

    def save(self, fname: str=False):
        """Write the contour to a file."""
        if not fname:
            fname = f"{self.parent_image.source_file}_{self.method}.txt"
        with open(fname, 'w') as f:
            json.dump({
                "method": self.method,
                "values": self.values.tolist()
            }, f)

    def plot(self, *args, **kwargs):
        """Plot the image with contours."""
        plot_contours(self, *args, **kwargs)
        if self.parent_image:
            plt.title(f"{self.parent_image.source_file} - Contour: {self.method}")
        else:
            plt.title(f"Contour: {self.method}")
        return

    def smooth(self, **kwargs):
        """Smooth the contour."""
        smooth_bumps(self, **kwargs)
        return

    def _load(self, file: str, image):
        """Load a contour from a file."""
        with open(file) as f:
            data = json.load(f)
        self.values = np.array(data["values"])
        self.method = data["method"]
        self.parent_image = image


class RippleImage:
    """Class for ripple images."""

    def __init__(self, *args, roi_x: list[int] = False, roi_y: list[int] = False):
        self.contours: list[RippleContour] = []
        # Handle loading from file if the file extension is .rimg
        if len(args) == 1 and isinstance(args[0], str) and str(args[0]).endswith('.rimg'):
            self._load(args[0])
            return

        if len(args) == 1:
            if isinstance(args[0], str) or isinstance(args[0], Path):
                file = args[0]
                if isinstance(args[0], Path):
                    file = str(file.resolve())
                self.image = load_image(file)
                self.source_file = file
            else:
                raise ValueError("Invalid input, expected a path to an image file or fname, image data pair.")
        elif len(args) == 2:
            if isinstance(args[0], str) and isinstance(args[1], np.ndarray):
                self.source_file = args[0]
                self.image = args[1]
            else:
                raise ValueError("Invalid input, expected a path to an image file or fname, image data pair.")
        self.image = preprocess_image(self.image, roi_x=roi_x, roi_y=roi_y)

    def __repr__(self) -> str:
        return f"RippleImage: {self.source_file.split('/')[-1]}"

    def add_contour(self, *args):
        """Add a contour to the RippleImage object."""
        contour = RippleContour(*args, image=self)
        self.contours.append(contour)

    def smooth_contours(self, **kwargs):
        """Smooth all the contours in the image."""
        self.contours = [contour.smooth(**kwargs) for contour in self.contours]
        return

    def plot(self, include_contours: bool = True, *args, **kwargs):
        """Plot the image with optional contours."""
        plot_image(self, include_contours=include_contours, *args, **kwargs)
        plt.title("RippleImage: " + self.source_file.split('/')[-1])
        return

    def save(self, fname: str = False, save_image_data: bool = False):
        """Save the image and contours to a file."""
        if not fname:
            fname = self.source_file.replace('.tif', '.rimg')

        # Save the metadata and contours
        data = {
            "source_file": self.source_file,
            "contours": [{
                "method": contour.method,
                "values": contour.values.tolist()
            } for contour in self.contours],
            "image": self.image if save_image_data else None
        }
        with gzip.open(fname, 'wb') as f:
            pickle.dump(data, f)
        return fname

    def _load(self, file: str):
        """Load the image and contours from a file."""
        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)

        self.source_file = data["source_file"]
        self.image = data.get("image", None)
        if self.image is not None:
            self.image = preprocess_image(self.image)

        self.contours = []
        for contour_data in data["contours"]:
            contour = RippleContour(
                np.array(contour_data["values"]),
                contour_data["method"],
                image=self
            )
            self.contours.append(contour)


class RippleImageSeries:
    """Class for a series of ripple images."""

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str) and str(args[0]).endswith('.rimgs'):
            self._load(args[0])
        elif len(args) == 1 and isinstance(args[0], list) and all(isinstance(img, RippleImage) for img in args[0]):
            self.images = args[0]
        else:
            raise ValueError("Invalid input, expected a file path to a .rimgs file or a list of RippleImage objects.")

    def __repr__(self) -> str:
        return f"RippleImageSeries: {len(self.images)} images"

    def animate(self, fig=plt.figure(figsize=(12,8)), fname: str = None, **kwargs):
        """Animate the images."""
        ani = FuncAnimation(fig, partial(self.update, **kwargs),
                                         frames=range(len(self.images)), interval=200, repeat=False)
        if fname:
            ani.save(fname, writer='ffmpeg')
        return ani

    def update(self, frame, **kwargs):
        plt.clf()
        self.images[frame].plot(**kwargs)

    def save(self, fname: str = False, save_image_data: bool = False):
        """Save the image series to a file."""
        if not fname:
            fname = 'image_series.rimgs'

        # Save the metadata and contours
        data = [image.source_file.replace('.tif', '.rimg') for image in self.images]
        with gzip.open(fname, 'wb') as f:
            pickle.dump(data, f)

        for image in self.images:
            if fname:
                image_fname = fname.split('/')[0:-1] + [image.source_file.split("/")[-1].replace('.tif', '.rimg')]
                image_fname = "/".join(image_fname)
            else:
                image_fname = None
            image.save(fname=image_fname, save_image_data=save_image_data)
        return fname

    def _load(self, file: str):
        """Load the image series from a file."""
        with gzip.open(file, 'rb') as f:
            image_files = pickle.load(f)
        base_path = Path(file).parent
        self.images = [RippleImage(str(base_path / image_file.split("/")[-1])) for image_file in image_files] # TODO Path ojets should be accepted by RippleImage, etc.
