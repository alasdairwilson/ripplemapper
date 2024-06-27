from pathlib import Path

__all__ = ['example_data', 'example_dir']

example_data = [Path(file) for file in Path(__file__).parent.glob('*') if file.is_file() and file.suffix == '.tif']

example_dir = str(example_data[0].parent)
