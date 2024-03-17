# !pip install llama-index faiss-cpu llama-index-vector-stores-faiss

import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

if __name__ == "__main__":
    import os

    # Instructions:
    # Run the script with the following command: python constrained_rag.py
    # Ensure to have the products directory in the same directory as this script
    # Ensure to have the OPENAI_API_KEY environment variable set

    assert os.getenv("OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY"

    # load document vectors
    documents = SimpleDirectoryReader("products/").load_data()

    # load faiss index
    d = 1536  # dimension of the vectors
    faiss_index = faiss.IndexFlatL2(d)

    # create vector store
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # initialize storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # create index
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # Configure retriever
    retriever = VectorIndexRetriever(index=index, similarity_top_k=1)

    QA_PROMPT_TMPL = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given only the context information and no prior knowledge, "
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
        "Otherwise, state: I cannot answer."
    )
    STRICT_QA_PROMPT = PromptTemplate(
        QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
    )

    # Configure response synthesizer
    response_synthesizer = get_response_synthesizer(
        structured_answer_filtering=True,
        response_mode="refine",
        text_qa_template=STRICT_QA_PROMPT,
    )

    # Assemble query engine
    safe_query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=response_synthesizer
    )

    # Execute query and evaluate response
    print(safe_query_engine.query("describe a summer dress with price"))
    print(safe_query_engine.query("describe a horse"))
