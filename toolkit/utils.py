
import os

def get_label_path(image_path):
    """Finds the corresponding label file for a given image file in a robust way."""
    parts = image_path.split(os.sep)
    try:
        # Find the last occurrence of 'images' and replace it with 'labels'
        idx = len(parts) - 1 - parts[::-1].index('images')
        parts[idx] = 'labels'
        label_path_base, _ = os.path.splitext(os.sep.join(parts))
        return label_path_base + '.txt'
    except ValueError:
        # Fallback if 'images' is not in the path, though less common
        return os.path.splitext(image_path)[0] + '.txt'
