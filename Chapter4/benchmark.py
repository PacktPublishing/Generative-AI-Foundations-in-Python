#!pip install openai langchain[llms] huggingface_hub pandas torch

import os
import pandas as pd
from langchain.llms import OpenAI, HuggingFaceHub
from langchain import LLMChain, PromptTemplate
from tqdm.auto import tqdm
import torch


def verify_gpu():
    return torch.cuda.is_available()


def load_data(file_path):
    return pd.read_csv(file_path)


def save_data(data, file_path):
    data.to_csv(file_path, index=False)


def generate_descriptions(llm, model_input_data, template):
    prompt = PromptTemplate(template=template, input_variables=["product_metadata"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    descriptions = []
    for data in tqdm(model_input_data):
        description = llm_chain.run(data)
        descriptions.append(description)
    return descriptions


if __name__ == "__main__":
    # Configuration
    config = {
        "openai_api_key": os.environ["OPENAI_API_KEY"],
        "huggingface_token": os.environ["HUGGINGFACEHUB_API_TOKEN"],
        "product_data_path": "data/product_data.csv",
        "test_data_path": "data/test_data.csv",
        "reference_data_path": "data/reference_data.csv",
        "openai_model_name": "gpt-3.5-turbo-instruct",
        "huggingface_repo_id": "EleutherAI/gpt-neo-2.7B",
        "prompt_template": """
        Write a creative product description for the following product: {product_metadata}
        """,
    }

    assert (
        config["openai_api_key"] is not None
    ), "OpenAI API Key is required, set the OPENAI_API_KEY environment variable."
    assert (
        config["huggingface_token"] is not None
    ), "HuggingFace API Token is required, set the HUGGINGFACEHUB_API_TOKEN environment variable."

    # Verify if GPU is available
    print("GPU Available:", verify_gpu())

    # Load data
    product_data = load_data(config["product_data_path"])

    # Data preparation
    test_data = product_data.sample(frac=0.2, random_state=42)
    reference_data = product_data.drop(test_data.index)

    save_data(test_data, config["test_data_path"])
    save_data(reference_data, config["reference_data_path"])

    reference_data = load_data(config["reference_data_path"])
    reference_descriptions = reference_data["product_description"].tolist()
    product_images = reference_data["product_image"].tolist()

    # Initialize models
    llm_gpt3 = OpenAI(
        model_name=config["openai_model_name"], temperature=0.9, max_tokens=256
    )
    llm_neo = HuggingFaceHub(
        repo_id=config["huggingface_repo_id"], model_kwargs={"temperature": 0.9}
    )

    # Generate descriptions
    gpt3_descriptions = generate_descriptions(
        llm_gpt3, reference_descriptions, config["prompt_template"]
    )
    gptneo_descriptions = generate_descriptions(
        llm_neo, reference_descriptions, config["prompt_template"]
    )

    # Save generated descriptions
    gpt3_descriptions_df = pd.DataFrame(
        {"product_description": gpt3_descriptions, "product_image": product_images}
    )
    gpt3_descriptions_df.to_csv("data/gpt3_descriptions.csv", index=False)

    gptneo_descriptions_df = pd.DataFrame(
        {"product_description": gptneo_descriptions, "product_image": product_images}
    )
    gptneo_descriptions_df.to_csv("data/gptneo_descriptions.csv", index=False)
