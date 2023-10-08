from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from preprocessing.image_utils import dynamic_resize


class ImagePreprocessor:
    def __init__(self, max_size: int = 1000):
        """
        Initialize the ImagePreprocessor class.

        Args:
            max_size: Maximum allowed size for the dynamically resized image.
        """
        self.original_shape = None
        self.max_size = max_size

    def preprocess(self, img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses the given image using histogram equalization followed by normalization.

        Args:
            img: Input PIL Image object.

        Returns:
            1D numpy array after preprocessing.
        """
        # Resize the image
        img = dynamic_resize(img, max_size=self.max_size)

        # Store the original shape
        self.original_shape = img.size[::-1]  # [height, width]

        # Convert PIL Image to numpy array.
        img_array = np.array(img)

        # 1. Histogram Equalization.
        equalized_img = cv2.equalizeHist(img_array)

        # 2. Normalization: Scale pixel values to the range [0, 1].
        normalized_img = equalized_img / 255.0

        # Flatten and return.
        return normalized_img.flatten()

    def reverse_preprocess(self, img_array: np.ndarray) -> Image.Image:
        """
        Reverses the preprocessing steps applied to the image.

        Args:
            img_array: 1D numpy array after preprocessing.

        Returns:
            PIL Image object.
        """
        assert self.original_shape is not None, "You must preprocess an image before reversing!"

        # Reshape the flattened image back to its original shape
        img_2d = img_array.reshape(self.original_shape)
        
        # Ensure all values are within the range [0, 1]
        clamped_img_2d = np.clip(img_2d, 0, 1)

        # Inverse normalization: convert [0, 1] range to [0, 255]
        inv_normalized_img = (clamped_img_2d * 255).astype(np.uint8)

        # Convert the numpy array back to a PIL Image
        return Image.fromarray(inv_normalized_img)
