"""
===================================
Creating a RippleImageSeries object
===================================

Ripplemapper can store a series of RippleImages in a RippleImageSeries object.
"""

from ripplemapper.io import load_dir_to_obj
from ripplemapper.classes import RippleImageSeries
from ripplemapper.data.example import example_dir

imgs = load_dir_to_obj(example_dir)
image_series = RippleImageSeries(imgs)

print(image_series)