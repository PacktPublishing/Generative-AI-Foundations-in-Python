# !pip install rouge sumeval nltk

import nltk
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sumeval.metrics.rouge import RougeCalculator
from nltk.translate.meteor_score import meteor_score


def evaluate(reference_descriptions: list, generated_descriptions: list) -> tuple:
    nltk.download("wordnet", quiet=True)

    # Calculating BLEU score
    bleu_scores = [
        sentence_bleu([ref], gen)
        for ref, gen in zip(reference_descriptions, generated_descriptions)
    ]
    average_bleu = sum(bleu_scores) / len(bleu_scores)

    # Calculating ROUGE score
    rouge = RougeCalculator()
    rouge_scores = [
        rouge.rouge_n(gen, ref, 2)
        for ref, gen in zip(reference_descriptions, generated_descriptions)
    ]
    average_rouge = sum(rouge_scores) / len(rouge_scores)

    # Calculating METEOR score
    meteor_scores = [
        meteor_score([ref.split()], gen.split())
        for ref, gen in zip(reference_descriptions, generated_descriptions)
    ]
    average_meteor = sum(meteor_scores) / len(meteor_scores)

    return average_bleu, average_rouge, average_meteor


if __name__ == "__main__":
    # Configuration
    config = {
        "reference_data_path": "data/reference_data.csv",  # Update paths if needed
        "generated_descriptions_gpt3_path": "data/gpt3_descriptions.csv",
        "generated_descriptions_neo_path": "data/gptneo_descriptions.csv",
    }

    # Load reference descriptions from CSV
    reference_data = pd.read_csv(config["reference_data_path"])
    reference_descriptions = reference_data[
        "product_description"
    ].tolist()  # Update 'description_column_name'

    # Load generated descriptions from CSV
    generated_descriptions_gpt3 = pd.read_csv(
        config["generated_descriptions_gpt3_path"]
    )["product_description"].tolist()
    generated_descriptions_neo = pd.read_csv(config["generated_descriptions_neo_path"])[
        "product_description"
    ].tolist()

    # Evaluate for GPT-3 descriptions
    avg_bleu_gpt3, avg_rouge_gpt3, avg_meteor_gpt3 = evaluate(
        reference_descriptions, generated_descriptions_gpt3
    )
    print(
        f"GPT-3: BLEU={avg_bleu_gpt3}, ROUGE={avg_rouge_gpt3}, METEOR={avg_meteor_gpt3}"
    )

    # Evaluate for GPT-Neo descriptions
    avg_bleu_neo, avg_rouge_neo, avg_meteor_neo = evaluate(
        reference_descriptions, generated_descriptions_neo
    )
    print(
        f"GPT-Neo: BLEU={avg_bleu_neo}, ROUGE={avg_rouge_neo}, METEOR={avg_meteor_neo}"
    )
