import os
from langchain_chroma import Chroma
from embeddings import get_embeddings


COLLECTION_NAME = "uniquery_gguf_q4km"


def storing_vectors(chunks, db_name: str):

    embeddings = get_embeddings()

    if os.path.exists(db_name):
        Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=db_name,
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        ids=[chunk.metadata["chunk_id"] for chunk in chunks],
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=db_name,
    )

    print(
        f"Vector Store of {vectorstore._collection.count()} "
        "vectors created successfully..."
    )

    return vectorstore
