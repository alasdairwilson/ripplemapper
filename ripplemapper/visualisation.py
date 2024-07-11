import warnings

import matplotlib.pyplot as plt
import numpy as np

__all__ = ['plot_contours', 'plot_image', 'plot_timeseries']

def plot_contours(ripple_contours, *args, **kwargs):
    """
    Plot the contour.

    Parameters
    ----------
    ripple_contours : RippleContour or list of RippleContour
        The contour or list of contours to plot.
    *args : tuple
        Additional positional arguments for the plot function.
    **kwargs : dict
        Additional keyword arguments for the plot function.
    """
    if not isinstance(ripple_contours, list):
        ripple_contours = [ripple_contours]
    for contour in ripple_contours:
        if contour.parent_image is not None:
            label = contour.parent_image.source_file.split('/')[-1] + ' : ' + contour.method
        else:
            label = contour.method
        plt.plot(contour.values[1], contour.values[0], label=label, *args, **kwargs)
    # set y-axis to match images (inverted)
    ax = plt.gca()
    ax.set_ylim((np.max(ax.get_ylim()), np.min(ax.get_ylim())))

def plot_image(ripple_image, include_contours: bool = True, cmap: str = 'gray', **kwargs):
    """
    Plot a RippleImage object.

    Parameters
    ----------
    ripple_image : RippleImage
        The RippleImage object to plot.
    include_contours : bool, optional
        Whether to include all the RippleContours on the plot, by default True.
    cmap : str, optional
        Colormap to use for plotting the image, by default 'gray'.
    **kwargs : dict
        Additional keyword arguments for the imshow function.
    """
    if ripple_image.image is None:
        if ripple_image.contours is None:
            raise ValueError("RippleImage object must have an image or contours to plot.")
        warnings.warn(f"Image not loaded for image: {ripple_image.source_file} plotting contours only.")
        x_max = y_max = 150
        x_min = y_min = np.inf
        for contour in ripple_image.contours:
            x_max = max(x_max, np.max(contour.values[1]))
            x_min = min(x_min, np.min(contour.values[1]))
            y_max = max(y_max, np.max(contour.values[0]))
            y_min = min(y_min, np.min(contour.values[0]))
        if x_min == np.inf:
            x_min = 0
        if y_min == np.inf:
            y_min = 0
        x_min, x_max, y_min, y_max = int(np.floor(x_min)), int(np.ceil(x_max)), int(np.floor(y_min)), int(np.ceil(y_max))
        plt.imshow(np.zeros((y_max, x_max)), cmap=cmap, **kwargs)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.gca().invert_yaxis()
    else:
        plt.imshow(ripple_image.image, cmap=cmap, **kwargs)
    if include_contours:
        if len(ripple_image.contours) == 0:
            warnings.warn(f"No contours found for image: {ripple_image.source_file} but you selected include_contours=True.")
        for contour in ripple_image.contours:
            plt.plot(contour.values[:][1], contour.values[:][0], label=contour.method)
        plt.legend()

def plot_timeseries(contours, labels, **kwargs):
    """
    Plot a timeseries of contours.

    Parameters
    ----------
    contours : list of RippleContour
        List of contours to plot.
    labels : list of str
        List of labels corresponding to each contour.
    **kwargs : dict
        Additional keyword arguments for the plot function.
    """
    for i, contour in enumerate(contours):
        plt.plot(contour.values[1], contour.values[0], label=labels[i], **kwargs)
    plt.gca().invert_yaxis()
    plt.legend()
