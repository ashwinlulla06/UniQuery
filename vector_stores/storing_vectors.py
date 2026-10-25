from langchain_chroma import Chroma

from config import resolve_project_path
from embeddings import get_embeddings


COLLECTION_NAME = "uniquery_gguf_q4km"


def storing_vectors(chunks, db_name: str):
    if not chunks:
        raise ValueError(
            "No chunks were supplied to the vector store."
        )

    db_path = resolve_project_path(db_name)
    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if db_path.exists():
        Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(db_path),
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        ids=[
            chunk.metadata["chunk_id"]
            for chunk in chunks
        ],
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(db_path),
    )

    print(
        f"Vector Store of "
        f"{vectorstore._collection.count()} "
        "vectors created successfully..."
    )

    return vectorstore
