from clip_score import *


if __name__ == "__main__":
    # Instructions:
    # Run the script using the command: python moderator.py

    # Define acceptable images
    acceptable_uploads = [
        "a detailed photo of a car",
        "an image of automotive parts",
        "cars racing on a track"
    ]

    # Load images
    images, paths = load_pil_images("./img/", prefix="car_example_image_car", format="jpeg", return_paths=True)
    # Load CLIP model
    model, processor = load_model_and_processor("openai/clip-vit-large-patch14")

    # Process image and text inputs together
    inputs = process_inputs(processor, acceptable_uploads, images)

    # Extract the probabilities
    probs = get_probabilities(model, inputs)

    # Display each image with corresponding scores
    display_images_with_scores(images, probs, names=paths)
