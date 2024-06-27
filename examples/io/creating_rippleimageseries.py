"""
===================================
Creating a RippleImageSeries object
===================================

Ripplemapper can store a series of RippleImages in a RippleImageSeries object.
"""

from ripplemapper.classes import RippleImageSeries
from ripplemapper.data.example import example_dir
from ripplemapper.io import load_dir_to_obj

imgs = load_dir_to_obj(example_dir)
image_series = RippleImageSeries(imgs)

print(image_series)
