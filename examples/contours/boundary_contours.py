"""
===========================================
Adding "Boundary" Contours to an Image
===========================================

One of the foundational 
"""

################################################################################
# We can load an image into a RippleImage object using the RippleImage classs and then run `add_boundary_contours` to add boundary contours to the image.

from ripplemapper.classes import RippleImage
from ripplemapper.data.example import example_data
from ripplemapper.analyse import add_boundary_contours
import matplotlib.pyplot as plt

ripple_img = RippleImage(example_data[-1])
add_boundary_contours(ripple_img, sigma=2)

################################################################################
# Plotting the image and its contours

ripple_img.plot(include_contours=True)
plt.show()