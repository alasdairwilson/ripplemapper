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
