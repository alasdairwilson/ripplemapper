"""
========================================================
Loading a directory of images into RippleImage objects
========================================================

Ripplemapper is capable of loading a directory of images into a list of RippleImage objects.
"""

from ripplemapper.data.example import example_dir
from ripplemapper.io import load_dir_to_obj

imgs = load_dir_to_obj(example_dir)
print(imgs)

#################################################################
#
# We can also select a subset of the images in our path with start, end and skip kwargs.
# These integer valued parameters work like python slicing [start:end:skip].

imgs = load_dir_to_obj(example_dir, start=0, end=3, skip=2)
print(imgs)

#################################################################
#
# This allows you to chunk up your input into independent processes or to quickly load a subset of your data for testing.
#
# Since you can save and load RippleImage objects, you can also save the subset of images.
# If this is done without the image arrays then you can fit far more images in memory at once
