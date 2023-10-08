import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from preprocessing.custom_preprocess import ImagePreprocessor


def sample_and_plot_images(image_objects: List[Image.Image], labels: np.ndarray, sample_size: int = 5):
    """
    Randomly sample images and plot them with their labels. Starts a new row for every 5 images.

    Args:
        image_objects: List of original Image objects.
        labels: A numpy array of labels.
        sample_size: Total number of images to sample and plot.
    """

    # Randomly select indices without replacement
    indices = np.random.choice(len(image_objects), sample_size, replace=False)

    sampled_images = [image_objects[i] for i in indices]
    sampled_labels = [labels[i] for i in indices]

    # Calculate number of rows and columns for the plot
    num_rows = -(-sample_size // 5)  # Ceiling division
    num_cols = min(sample_size, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))

    # Ensure axes is a 2D array for easy indexing
    if num_rows == 1 or num_cols == 1:
        axes = np.array(axes).reshape(num_rows, num_cols)

    for i, (img, lbl) in enumerate(zip(sampled_images, sampled_labels)):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # Hide axes
        ax.set_title(f"Label: {lbl}")

    plt.tight_layout()
    plt.show()


def plot_reconstructed_images(reconstructed_matrix: np.ndarray,
                              original_images: List[Image.Image],
                              preprocessor: 'ImagePreprocessor',
                              num_samples: int = 5,
                              name: str = "reconstructed_image_plot.png",
                              save_only: bool = True) -> str:
    """
    Plot pairs of original and reconstructed images side by side and save the plot as an image.

    Args:
        reconstructed_matrix (np.ndarray): The matrix containing the reconstructed images' data.
            Each row represents a flattened image.
        original_images (List[Image.Image]): A list of original PIL Image objects.
        preprocessor (ImagePreprocessor): An instance of the ImagePreprocessor class to reverse preprocess images.
        num_samples (int, optional): Number of random image pairs to sample and plot. Default is 5.
        name (str, optional): Name of the file where the plot will be saved. Default is "reconstructed_image_plot.png".
        save_only (bool, optional): If True, the plot will be saved and not shown. If False, the plot will also be shown

    Returns:
        str: The path where the image plot was saved.
    """
    # Sample random indices for visualization
    indices = np.random.choice(reconstructed_matrix.shape[0], num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))

    for idx, ax in zip(indices, axes):
        # Original images
        ax[0].imshow(original_images[idx], cmap='gray')
        ax[0].set_title(f'Original Image {idx}')
        ax[0].axis('off')

        # Reconstructed images
        reconstructed_img_obj = preprocessor.reverse_preprocess(reconstructed_matrix[idx, :])
        ax[1].imshow(reconstructed_img_obj, cmap='gray')
        ax[1].set_title(f'Reconstructed Image {idx}')
        ax[1].axis('off')

    plt.tight_layout()

    # Ensure the directory exists
    directory = './images/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = os.path.join(directory, name)
    plt.savefig(save_path)

    if not save_only:
        plt.show()

    plt.close()  # Close the pyplot figure to free memory

    return save_path
