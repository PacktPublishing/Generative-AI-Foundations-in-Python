import torch
from PIL import Image
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple

# Constants and Configuration
config = {
    "clip_repo": "openai/clip-vit-base-patch32",
    "reference_data_path": "data/reference_data.csv",
    "generated_descriptions_gpt3_path": "data/gpt3_descriptions.csv",
    "generated_descriptions_neo_path": "data/gptneo_descriptions.csv",
    "image_column_name": "product_image",  # Update as necessary
    "description_column_name": "product_description",  # Update as necessary
}


def load_image_from_path(image_path: str, crop_size=(300, 300)) -> Image.Image:
    try:
        with Image.open(image_path) as img:
            img.load()
            width, height = img.size
            left = (width - crop_size[0]) / 2
            top = (height - crop_size[1]) / 2
            right = (width + crop_size[0]) / 2
            bottom = (height + crop_size[1]) / 2
            img_cropped = img.crop((left, top, right, bottom))
            return img_cropped
    except IOError as error:
        print(f"Error opening or loading the image file: {error}")
        return None


def load_model_and_processor(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def process_inputs(
    processor: CLIPProcessor, prompts: List[str], images: List[Image.Image]
) -> dict:
    processed_texts = processor(
        text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    processed_images = processor(images=images, return_tensors="pt")
    return {
        "input_ids": processed_texts["input_ids"],
        "attention_mask": processed_texts["attention_mask"],
        "pixel_values": processed_images["pixel_values"],
    }


def clip_scores(images, descriptions, model, processor) -> List[float]:
    scores = []
    inputs = process_inputs(processor, descriptions, images)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    for i in range(logits_per_image.size(0)):
        score = logits_per_image[i, i].item()
        scores.append(score)
    return scores


if __name__ == "__main__":
    clip_model, clip_processor = load_model_and_processor(config["clip_repo"])
    reference_data = pd.read_csv(config["reference_data_path"])

    reference_images = [
        load_image_from_path(row[config["image_column_name"]])
        for _, row in reference_data.iterrows()
        if row[config["image_column_name"]] is not None
    ]

    gpt3_descriptions = pd.read_csv(config["generated_descriptions_gpt3_path"])[
        config["description_column_name"]
    ].tolist()
    gptneo_descriptions = pd.read_csv(config["generated_descriptions_neo_path"])[
        config["description_column_name"]
    ].tolist()
    reference_descriptions = reference_data[config["description_column_name"]].tolist()

    gpt3_generated_scores = clip_scores(
        reference_images, gpt3_descriptions, clip_model, clip_processor
    )
    gptneo_generated_scores = clip_scores(
        reference_images, gptneo_descriptions, clip_model, clip_processor
    )
    reference_scores = clip_scores(
        reference_images, reference_descriptions, clip_model, clip_processor
    )

    # Example usage of printing scores
    print(f"GPT-3 Generated Scores: {gpt3_generated_scores}")
    print(f"GPT-Neo Generated Scores: {gptneo_generated_scores}")
    print(f"Reference Scores: {reference_scores}")
