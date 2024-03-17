# !pip install pytorch-fid torch diffusers clip transformers accelerate matplotlib

from typing import List
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline


def load_model(model_id: str, device="cpu") -> StableDiffusionPipeline:
    """Load model with provided model_id."""
    return StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=False
    ).to(device)


def generate_images(
    pipe: StableDiffusionPipeline, prompts: List[str], device="cuda"
) -> torch.Tensor:
    """Generate images based on provided prompts."""
    with torch.autocast(device):
        images = pipe(prompts).images
    return images


def render_images(images: torch.Tensor):
    """Plot the generated images."""
    plt.figure(figsize=(10, 5))
    for i, img in enumerate(images):
        plt.subplot(1, 2, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Instructions:
    # Replace the model_id with your choice.
    # Add the desired prompts to the prompts list.
    # Run the script using the command: python stable_diff.py

    model_id = "CompVis/stable-diffusion-v1-4"
    prompts = [
        "a hyper-realistic photo of a modern sneaker",
        "A stylized t-shirt with an sports-inspired design",
    ]

    device = "mps"  # "cuda", "cpu", "mps" is for M1 Macs
    pipe = load_model(model_id, device="mps")
    images = generate_images(pipe, prompts, device="cpu") # autocast does not support mps
    render_images(images)
