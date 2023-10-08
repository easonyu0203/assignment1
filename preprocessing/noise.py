import numpy as np
from PIL import Image, ImageFilter


class BlockNoise:
    def __init__(self, prob: float = 0.5, mean: float = 0.1, std: float = 0.05):
        """
        Initialize the BlockNoise class with default parameters for block noise.

        Args:
            prob: Probability of applying block noise.
            mean: Mean percentage of the image width for block width.
            std: Standard deviation percentage of the image width for block width.
        """
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies block noise to the given image based on instance parameters.

        Args:
            img: The input PIL Image object.

        Returns:
            The image with block noise (if applied).
        """
        if np.random.rand() > self.prob:
            return img

        img_width, img_height = img.size
        block_size_percentage = np.random.normal(self.mean, self.std)
        block_size = int(img_width * block_size_percentage)
        block_size = max(1, min(block_size, img_width))

        start_x = np.random.randint(0, img_width - block_size)
        start_y = np.random.randint(0, img_height - block_size)

        img_array = np.array(img)
        img_array[start_y:start_y + block_size, start_x:start_x + block_size] = 255

        return Image.fromarray(img_array)


class BlurNoise:
    def __init__(self, prob: float = 0.5, mean: float = 2.0, std: float = 1.0):
        """
        Initialize the BlurNoise class with default parameters for blur noise.

        Args:
            prob: Probability of applying the blur.
            mean: The mean of the Gaussian distribution for blur radius.
            std: The standard deviation of the Gaussian distribution for blur radius.
        """
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply blur to the given image based on instance parameters.

        Args:
            img: Input PIL Image object.

        Returns:
            Blurred (or original) PIL Image object.
        """
        if np.random.rand() < self.prob:
            blur_radius = max(0.1, np.random.normal(self.mean, self.std))
            return img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return img
