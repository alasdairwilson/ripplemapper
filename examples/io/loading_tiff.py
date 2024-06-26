"""
====================
Loading a Tiff Image
====================

This example shows how to load a tiff file into a numpy array.
"""

from ripplemapper.io import load_image
from ripplemapper.data.example import example_data
from matplotlib import pyplot as plt

img = load_image(example_data[-1])

plt.imshow(img)
plt.show()

