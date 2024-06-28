"""
========================
Plotting a ripple image
========================

How to plot a ripple image using inbuilt plotting methods.
"""

from ripplemapper.classes import RippleImage
from ripplemapper.data.example import example_data
from ripplemapper.analyse import add_boundary_contours

ripple_img = RippleImage(example_data[2])
add_boundary_contours(ripple_img, sigma=2)

#################################################################
#
# We can plot the image using the `RippleImage.plot' object, or by calling the `plot_ripple_image' function from the `ripplemapper.visualisation'.

ripple_img.plot(include_contours=False)

#################################################################
# If there are contours on the image, we can plot the contours

ripple_img.plot(include_contours=True)