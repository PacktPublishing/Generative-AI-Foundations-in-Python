# !pip install ragas tqdm llama-index faiss-cpu llama-index-vector-stores-faiss

import os
import faiss
from datasets import Dataset
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Response,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness,
)
from ragas.metrics.critique import harmfulness
from tqdm.auto import tqdm
from typing import List, Dict, Any, Callable

# ensure our API key is set
assert os.getenv("OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY"


def load_index(dir_path: str = "products/", dim: int = 1536) -> VectorStoreIndex:
    """Load the index from the given directory path"""
    documents = SimpleDirectoryReader(dir_path).load_data()
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index


def main() -> None:
    # Load the index and create a query engine
    index: VectorStoreIndex = load_index()
    query_engine = index.as_query_engine()

    # Define the questions to ask the model
    questions: List[str] = [
        "What features does the Chic Summer Dress offer?",
        "How much does the Urban Streetwear Hoodie cost?",
        "What material is the Sleek Leather Jacket made of?",
        "What are the key characteristics of the Vintage High-Waisted Jeans?",
    ]

    # Query the model and get the responses
    response_objects: List[Response] = []
    for q in tqdm(questions):
        response: Response = query_engine.query(q)
        response_objects.append(response)

    # Extract the responses from the response objects
    engine_responses: List[str] = [r.response for r in response_objects]

    # Define the evaluation data
    eval_data: Dict[str, Any] = {
        "question": questions,
        "answer": engine_responses,
        "contexts": [
            [
                "A lightweight summer dress with a vibrant floral print, perfect for sunny days."
            ],
            [
                "An edgy hoodie featuring a bold graphic design, complete with a cozy kangaroo pocket."
            ],
            [
                "A sleek leather jacket that offers a slim fit and stylish zippered pockets for the modern urban look."
            ],
            [
                "High-waisted jeans with just the right amount of stretch and distressed details for a vintage vibe."
            ],
        ],
        "ground_truth": [
            "A Chic Summer Dress that features a lightweight fabric with a vibrant floral print and a knee-length cut, perfect for sunny days.",
            "The price of the Urban Streetwear Hoodie, which has an adjustable hood and a kangaroo pocket with a bold graphic design, is $79.99.",
            "A Sleek Leather Jacket made of genuine leather, featuring zippered pockets and a slim fit for the modern urban look.",
            "Vintage High-Waisted Jeans which are high-waisted with distressed details and made of stretch denim, embodying a vintage vibe.",
        ],
    }
    # Create a dataset from the evaluation data
    dataset: Dataset = Dataset.from_dict(eval_data)

    # Define the evaluation metrics
    metrics: List[Callable] = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy,
        harmfulness,
    ]

    # Evaluate the model using the defined metrics
    result: Dict[str, float] = evaluate(dataset, metrics=metrics)
    print(result)


if __name__ == "__main__":
    main()
