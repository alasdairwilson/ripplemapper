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

################################################################################
#
# Just like with RippleImages we can select a Region of Interest when loading the image files.

imgs_with_roi = load_dir_to_obj(example_dir, roi_x=[0, 100], roi_y=[0, 100])
print(imgs[0].image.shape, imgs_with_roi[0].image.shape)
