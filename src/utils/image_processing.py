"""Image preprocessing utilities.

Common image operations used across the pipeline.
"""
import numpy as np
from PIL import Image


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is in RGB format with 3 channels.

    Handles:
    - Grayscale (H, W) -> (H, W, 3)
    - RGBA (H, W, 4) -> (H, W, 3)
    - Already RGB -> no-op

    Args:
        image: numpy array

    Returns:
        numpy array (H, W, 3) uint8
    """
    if len(image.shape) == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def crop_with_bounds_check(image: np.ndarray, bbox: list,
                           min_size: int = 10) -> np.ndarray:
    """Crop image with bounds checking.

    Args:
        image: numpy array (H, W, 3)
        bbox: [x1, y1, x2, y2] pixel coordinates
        min_size: minimum crop dimension to return valid result

    Returns:
        Cropped numpy array, or None if crop is invalid/too small
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < min_size:
        return None

    return crop


def resize_with_padding(image: np.ndarray, target_size: int = 640,
                        pad_value: int = 114) -> np.ndarray:
    """Resize image with letterbox padding to maintain aspect ratio.

    Args:
        image: numpy array (H, W, 3)
        target_size: target dimension (square)
        pad_value: padding pixel value

    Returns:
        numpy array (target_size, target_size, 3)
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    resized = np.array(resized)

    padded = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top:top + new_h, left:left + new_w] = resized

    return padded
