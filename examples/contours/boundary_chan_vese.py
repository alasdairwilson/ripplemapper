"""
=====================================
Adding Chan-Vese Contours to an Image
=====================================

This example shows how to add a contour based on the chan-vese segmentation of the image.

The Chan-vese algorithm is a level set method for image segmentation.
It is based on the Mumford-Shah functional and is designed to segment images without edges.

In this example, we will the `ripplemapper.analyse` module to segment the image and add the resulting contour to the image.
"""

################################################################################
# We can load an image into a RippleImage object using the RippleImage class and then run `add_chan_vese_contours` to add Chan-Vese contours to the image.

import matplotlib.pyplot as plt

from ripplemapper.analyse import add_chan_vese_contours
from ripplemapper.classes import RippleImage
from ripplemapper.data.example import example_data

ripple_img = RippleImage(example_data[1])
add_chan_vese_contours(ripple_img)

################################################################################
# Plotting the image and its contours

ripple_img.plot(include_contours=True)
plt.show()
