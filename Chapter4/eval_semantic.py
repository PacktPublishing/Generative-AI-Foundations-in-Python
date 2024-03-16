# Install required packages
# !pip install sentence-transformers

import pandas as pd
from sentence_transformers import SentenceTransformer, util


def cosine_similarity(model, reference_descriptions, generated_descriptions):
    # Calculating cosine similarity for generated descriptions
    cosine_scores = [
        util.pytorch_cos_sim(model.encode(ref), model.encode(gen))[0][0]
        for ref, gen in zip(reference_descriptions, generated_descriptions)
    ]
    average_cosine = sum(cosine_scores) / len(cosine_scores)
    return average_cosine


if __name__ == "__main__":
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Configuration
    config = {
        "reference_data_path": "data/reference_data.csv",
        "generated_descriptions_gpt3_path": "data/gpt3_descriptions.csv",
        "generated_descriptions_neo_path": "data/gptneo_descriptions.csv",
    }

    # Load reference descriptions from CSV
    reference_data = pd.read_csv(config["reference_data_path"])
    reference_descriptions = reference_data["product_description"].tolist()

    # Load generated descriptions from CSV
    generated_descriptions_gpt3 = pd.read_csv(
        config["generated_descriptions_gpt3_path"]
    )["product_description"].tolist()
    generated_descriptions_neo = pd.read_csv(config["generated_descriptions_neo_path"])[
        "product_description"
    ].tolist()

    # Evaluate cosine similarity
    average_cosine_gpt3 = cosine_similarity(
        model, reference_descriptions, generated_descriptions_gpt3
    )
    print(f"Average Cosine Similarity GPT-3: {average_cosine_gpt3}")

    average_cosine_neo = cosine_similarity(
        model, reference_descriptions, generated_descriptions_neo
    )
    print(f"Average Cosine Similarity GPT-Neo: {average_cosine_neo}")
