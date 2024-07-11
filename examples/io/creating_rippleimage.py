"""
===========================================
Loading an image into a RippleImage object
===========================================

One of the foundational classes of Ripplemapper is its image class, this example shows how to load an image file into a RippleImage object.
"""

from ripplemapper.classes import RippleImage
from ripplemapper.data.example import example_data

ripple_img = RippleImage(example_data[-1])
print(ripple_img)

################################################################################
# We can optionally select a region of interest via roi_x and roi_y keyword arguments, to cut out unnecessary parts of the image.
# These keywords are consistent in the directory loading methods.

print(ripple_img.image.shape)
ripple_img = RippleImage(example_data[-1], roi_x=[0, 100], roi_y=[0, 100])
print(ripple_img.image.shape)
