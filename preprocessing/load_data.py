import os
from typing import Tuple, List, Callable, Optional, Union
import numpy as np
from PIL import Image


def load_image_data(data_path: str,
                    preprocess_function: Optional[Callable[[Image.Image], np.ndarray]] = None,
                    noise_functions: Optional[List[Callable[[Image.Image], Image.Image]]] = None,
                    data_fraction: float = 1.0
                    ) -> Tuple[np.ndarray, np.ndarray, List[Image.Image]]:
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        data_path: Path to dataset.
        preprocess_function: Function to be applied to the image for preprocessing, should return a 1D numpy array.
        noise_functions: List of functions to introduce noise to the image, each should return an Image object.
        data_fraction: Fraction of data to be used. Should be between 0 and 1. Default is 1.0 which means use all data.

    Returns:
        image_data: A numpy array of preprocessed images.
        image_labels: A numpy array of labels for each image.
        image_objects: A list of original Image objects for future plotting.
    """
    image_data: List[np.ndarray] = []
    image_labels: List[int] = []
    image_objects: List[Image.Image] = []

    for i, person in enumerate(sorted(os.listdir(data_path))):
        person_path = os.path.join(data_path, person)

        if not os.path.isdir(person_path):
            continue

        for fname in os.listdir(person_path):
            if fname.endswith('Ambient.pgm') or not fname.endswith('.pgm'):
                continue

            # Load image.
            img_path = os.path.join(person_path, fname)
            img = Image.open(img_path).convert('L')  # Convert to greyscale.

            # Apply each noise function sequentially if provided.
            if noise_functions:
                for noise_fn in noise_functions:
                    img = noise_fn(img)

            # Save raw image for future plotting
            image_objects.append(img.copy())

            # Preprocess image
            if preprocess_function:
                x_array = preprocess_function(img)
            else:
                x_array = np.array(img).flatten()

            # Collect data and label.
            image_data.append(x_array)
            image_labels.append(i)

    # Randomly select a fraction of the data
    total_data = len(image_data)
    sampled_indices = np.random.choice(total_data, int(total_data * data_fraction), replace=False)

    image_data = np.vstack([image_data[i] for i in sampled_indices])
    image_labels = np.array([image_labels[i] for i in sampled_indices])
    image_objects = [image_objects[i] for i in sampled_indices]

    return image_data, image_labels, image_objects
