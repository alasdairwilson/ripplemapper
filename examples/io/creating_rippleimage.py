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
