from typing import List, Tuple
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from util import render_images, render_image, load_pil_images, DEMO_PROMPTS


def load_model_and_processor(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Loads the CLIP model and processor.
    """
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def process_inputs(
    processor: CLIPProcessor, prompts: List[str], images: List[Image.Image]
) -> dict:
    """
    Processes the inputs using the CLIP processor.
    """
    return processor(text=prompts, images=images, return_tensors="pt", padding=True)


def get_probabilities(model: CLIPModel, inputs: dict) -> torch.Tensor:
    """
    Computes the probabilities using the CLIP model.
    """
    outputs = model(**inputs)
    logits = outputs.logits_per_image

    # Define temperature -  higher temperature will make the distribution more uniform.
    T = 10

    # Apply temperature to the logits
    temp_adjusted_logits = logits / T

    probs = torch.nn.functional.softmax(temp_adjusted_logits, dim=1)

    return probs


def display_images_with_scores(
    images: List[Image.Image], scores: torch.Tensor, notebook:bool=False, names: list=[]
) -> None:
    """
    Displays the images alongside their scores.
    """
    # Set print options for readability
    torch.set_printoptions(precision=2, sci_mode=False)

    # Display the images and scores
    for i, image in enumerate(images):
        name = "Image" if not names else names[i]
        print(f"{name} {i + 1}:")
        if notebook:
            render_image(image)
        print(f"Scores: {scores[i, :].detach().numpy()}")
        print()


if __name__ == "__main__":
    # Instructions:
    # Run the script using the command: python clip_score.py
    # Images are loaded from the img directory.
    # The prompts are predefined in the DEMO_PROMPTS list in util.py.

    # Define prompts
    prompts = DEMO_PROMPTS

    # Load images
    images = load_pil_images("./img/", prefix="pil_image")

    # Load CLIP model
    model, processor = load_model_and_processor("openai/clip-vit-base-patch32")
   
    # Process image and text inputs together
    inputs = process_inputs(processor, prompts, images)
   
    # Extract the probabilities
    probs = get_probabilities(model, inputs)
   
    # Display each image with corresponding scores
    display_images_with_scores(images, probs)
