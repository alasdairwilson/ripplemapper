"""
=========================================================
Loading an directory of images into a RippleImage objets
=========================================================

Ripplemapper is capable of loading a directory of images into a list of RippleImage objects.
"""

from ripplemapper.io import load_dir_to_obj
from ripplemapper.data.example import example_dir

imgs = load_dir_to_obj(example_dir)

print(imgs)