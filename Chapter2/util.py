import matplotlib.pyplot as plt
import torch
import fnmatch
from PIL import Image
import os

DEMO_PROMPTS = [
    "a hyper-realistic photo of a modern sneaker",
    "A stylized t-shirt with an sports-inspired design",
]

def render_image(image: torch.Tensor):
    """Plot the generated image."""
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def render_images(images: torch.Tensor):
    """Plot the generated images."""
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(images):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()


def save_pil_images(images, directory, prefix="image", format="png"):
    """
    Saves a list of PIL images to the specified directory.

    Parameters:
    - images: a list of PIL Image objects.
    - directory: path to the directory where images will be saved.
    - prefix: prefix for the saved image filenames.
    - format: format of the saved images ('png', 'jpg', etc.).
    """

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterate through the list of images and save each one
    for i, img in enumerate(images):
        img.save(os.path.join(directory, f"{prefix}_{i}.{format}"))


def load_pil_images(directory, prefix="image", format="png", return_paths=False):
    """
    Loads PIL images from a specified directory into a list.

    Parameters:
    - directory: path to the directory from which images will be loaded.
    - prefix: prefix for the filenames of images to be loaded.
    - format: format of the images to be loaded ('png', 'jpg', etc.).

    Returns:
    - A list of PIL Image objects.
    """

    images = []
    img_paths = []
    # Construct the full path pattern to filter images by prefix and format
    file_pattern = f"{prefix}_*.{format}"

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, file_pattern):
            img_path = os.path.join(directory, filename)
            img_paths.append(img_path)
            try:
                with Image.open(img_path) as img:
                    images.append(img.copy())  # Copy image to avoid closing
            except IOError:
                print(f"Error loading image: {img_path}")

    if return_paths:
        return images, img_paths

    return images
