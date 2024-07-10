import os

import matplotlib.pyplot as plt
import pytest

from ripplemapper.classes import RippleContour


def test_ripple_contour_to_physical(loaded_example_contour):
    # Assuming the function is not yet implemented
    loaded_example_contour.to_physical()

def test_ripple_contour_save(loaded_example_contour, tmp_path):
    save_path = tmp_path / "contour.txt"
    loaded_example_contour.save(str(save_path))
    assert os.path.exists(save_path)

def test_ripple_contour_plot(loaded_example_contour):
    loaded_example_contour.plot()
    plt.close()

def test_ripple_contour_smooth(loaded_example_contour):
    loaded_example_contour.smooth()
    # Assuming the smooth function does not return anything but modifies in place
    assert loaded_example_contour.values is not None

def test_ripple_contour_load(loaded_example_contour, tmp_path):
    save_path = tmp_path / "contour.txt"
    loaded_example_contour.save(str(save_path))
    new_contour = RippleContour(str(save_path))
    assert new_contour.values is not None
    assert new_contour.method == loaded_example_contour.method

def test_ripple_image_series_save(loaded_example_image_series, tmp_path):
    save_path = tmp_path / "image_series.rimgs"
    loaded_example_image_series.save(str(save_path))
    assert os.path.exists(save_path)

@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything.")
def test_ripple_image_series_animate(loaded_example_image_series):
    ani = loaded_example_image_series.animate()
    assert ani is not None

@pytest.mark.filterwarnings("ignore:Image not loaded for image")
@pytest.mark.filterwarnings("ignore:No contours found for image")
def test_ripple_image_series_update(loaded_example_image_series):
    plt.figure()
    loaded_example_image_series.update(0)
    plt.close()

if __name__ == '__main__':
    pytest.main()
