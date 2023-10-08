import cv2
import numpy as np
import math
from PIL import Image


def dynamic_resize(img: Image.Image, max_size: int) -> Image.Image:
    """
    Dynamically resizes an image so that the flattened feature array is as large as possible but below max_size,
    while maintaining its aspect ratio.

    Args:
        img: Input PIL Image object.
        max_size: Maximum allowed size of the feature array after flattening.

    Returns:
        Resized PIL Image object.
    """
    # Calculate original aspect ratio
    aspect_ratio = img.width / img.height

    # Calculate new dimensions based on the max_size
    new_height = int(math.sqrt(max_size / aspect_ratio))
    new_width = int(aspect_ratio * new_height)

    # Ensure the flattened size does not exceed max_size
    while new_width * new_height > max_size:
        new_height -= 1
        new_width = int(aspect_ratio * new_height)

    # Resize the image
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    return resized_img
