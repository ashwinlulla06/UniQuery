from langchain_chroma import Chroma
from embeddings import get_embeddings
from vector_stores.storing_vectors import COLLECTION_NAME
import config


RETRIEVAL_K = 15
DB_NAME = config.DB_NAME

def fetch_documents(question) -> list:

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_NAME,
        embedding_function=get_embeddings(),
    )

    if vector_store._collection.count() == 0:
        raise ValueError(
            f"Vector store '{DB_NAME}' is empty. Create it before running retrieval."
        )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )

    return retriever.invoke(question)
