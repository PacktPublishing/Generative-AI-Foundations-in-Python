# pip install pillow
from PIL import Image
import os


def resize_images(folder_path: str, new_size: tuple = (800, 600)) -> None:
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, filename)
            with Image.open(img_path) as img:
                resized_img = img.resize(new_size)
                # Save the resized image to the same location
                resized_img.save(img_path)
                print(f"Resized {filename}")


if __name__ == "__main__":
    # Replace 'path_to_images' with the path to the folder containing the images
    path_to_images = os.getcwd() + "/chap-four/data/img/"
    resize_images(path_to_images)
