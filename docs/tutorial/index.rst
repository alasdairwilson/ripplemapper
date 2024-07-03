.. _tutorial:


The `ripplemapper` Tutorial
============================

This tutorial will guide you through the basic usage of `ripplemapper`.

.. contents:: Tutorial
    :depth: 2
    :backlinks: none

Installation
************

To install `ripplemapper`, you can use `pip`:

.. code-block:: bash

    pip install ripplemapper

Loading an image
*****************

`ripplemapper.classes.RippleImage` is the main class for handling images in `ripplemapper`.

You can load a "tiff" file into a `RippleImage` object via its constructor:

.. code-block:: python

    from ripplemapper.classes import RippleImage

    image = RippleImage("ripplemapper/data/example/1_00375_sample.tif")

We can see the image data in `RippleImage.data`, and visualise it with the `RippleImage.plot` method.

.. code-block:: python

    image.data
    image.plot()
    plt.show()

The `ripplemapper.io` module provides alternative functions for loading data, either individually or from a directory.

.. code-block:: python

    from ripplemapper.io import load_image, load_dir_to_obj

    image = load_image("ripplemapper/data/example/1_00375_sample.tif")
    images = load_dir_to_obj("ripplemapper/data/example")

There is also a `RippleImageSeries` class for handling multiple images.

.. code-block:: python

    from ripplemapper.classes import RippleImageSeries

    series = RippleImageSeries(images)
    print(series)

Analysing the images
********************

The basic principle this package operates is a variety of functions that use the imaeg data to produce contours across the domain.
These contours should hopefully correspond to the breaking wave interface in the image.

These functions are generally stored in the `ripplemapper.contour` module.

The `ripplemapper.analyse` module provides helper functions for using these on `RippleImage` objects.

.. code-block:: python

    from ripplemapper.analyse import add_boundary_contours, add_chan_vese_contours, add_a_star_contours

    add_boundary_contours(image)
    add_chan_vese_contours(image)
    add_a_star_contours(image)

    image.plot()
    # Now the various contours are visible in our image
    plt.show()

All of these functions also work with `RippleImageSeries` objects, e.g.

.. code-block:: python

    add_boundary_contours(series, overwrite=True)

    series.images[0].plot()
    plt.show()

For a more detailed explanation of how to use these functions, see the :ref:`reference` section.

Saving and loading data
************************

Once you have finished, you can save the data to a file.

You can save the `RippleImage` to a ".rimg" file using the `RippleImage.save` method, by default this does not actually include the image data, but this can be included via the `save_image_data` argument.
This will always save all the contours stored in the `RippleImage.contours` attribute.

.. code-block:: python

    image.save("example.rimg", save_image_data=True)

Individual contours can be saved with the `RippleContour.save` method.

.. code-block:: python

    for contour in image.contours:
        contour.save(fname = f"example_contour_{contour.method}.rcontour")

A `RippleImageSeries` can be saved to a directory, as both a ".rimgs" file and a ".rimg" file for each image.

.. code-block:: python

    series.save("example_series.rimgs")

Any of these files can be loaded via their corresponding class constructor or via the `ripplemapper.io.load` function.
