# !pip install llama-index faiss-cpu llama-index-vector-stores-faiss

import faiss

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.vector_stores.faiss import FaissVectorStore
# from IPython.display import Markdown, display

if __name__ == "__main__":
    import os

    assert os.getenv("OPENAI_API_KEY") is not None, "Please set OPENAI_API_KEY"

    # load document vectors
    documents = SimpleDirectoryReader("products/").load_data()

    # load faiss index
    d = 1536 # dimension of the vectors
    faiss_index = faiss.IndexFlatL2(d)

    # create vector store
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    # initialize storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # create index
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

    # query the index
    query_engine = index.as_query_engine()
    response = query_engine.query("describe summer dress with price")

    print(response)
    # display(Markdown(f"<b>{response}</b>"))
