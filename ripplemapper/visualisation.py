import warnings

import matplotlib.pyplot as plt
import numpy as np


def plot_contours(ripple_contours, *args, **kwargs):
    """Plot the contour."""
    if not isinstance(ripple_contours, list):
        ripple_contours = [ripple_contours]
    for contour in ripple_contours:
        label = contour.parent_image.source_file.split('/')[-1] + ' : ' + contour.method
        plt.plot(contour.values[1], contour.values[0],  label=label, *args, **kwargs)
    # set y axis to be high to low
    ax = plt.gca()
    ax.set_ylim((np.max(ax.get_ylim()), np.min(ax.get_ylim())))

def plot_image(ripple_image, include_contours: bool=True, *args, **kwargs):
    """Plot a RippleImage object.

    Parameters
    ----------
    ripple_image : RippleImage
        The RippleImage object to plot.
    include_contours : bool, optional
        whether to include all the RippleContours on the plot, by default True
    """
    plt.imshow(ripple_image.image, cmap='gray')
    if include_contours:
        if len(ripple_image.contours) == 0:
            warnings.warn(f"No contours found for image: {ripple_image.source_file} but you selected include_contours=True.")
        for contour in ripple_image.contours:
            plt.plot(contour.values[:][1], contour.values[:][0], *args, **kwargs, label=contour.method)
    plt.legend()
