from pathlib import Path

__all__ = ['example_data', 'example_dir', 'example_contour', 'example_rimgs']

example_data = sorted([Path(file) for file in Path(__file__).parent.glob('*') if file.is_file() and file.suffix == '.tif'])

example_dir = str(example_data[0].parent)

example_contour = str((example_data[0].parent / 'example_contour.txt').resolve())

example_rimgs = str((example_data[0].parent / 'rimgs' /  'image_series.rimgs').resolve())
