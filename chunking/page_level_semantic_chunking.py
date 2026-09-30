import json

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from config import DOCUMENT_CACHE_PATH
from embeddings import get_embeddings


def load_cached_documents():
    documents = []

    with DOCUMENT_CACHE_PATH.open(
        "r",
        encoding="utf-8",
    ) as cache_file:
        for line in cache_file:
            record = json.loads(line)
            documents.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"],
                )
            )

    return documents


def page_level_chunking():
    text_splitter = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )

    documents = load_cached_documents()

    if not documents:
        print("No documents were extracted.")
        return []

    chunks = text_splitter.split_documents(documents)

    print("Number of Chunks: ", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['file_name']}__chunk-{index}"
        )

    return chunks
