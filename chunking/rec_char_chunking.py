import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DOCUMENT_CACHE_PATH


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


def rec_char_chunking(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = load_cached_documents()

    if not documents:
        print("No documents found!")
        return []

    chunks = text_splitter.split_documents(documents)

    print("Number of Chunks: ", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata['file_name']}__chunk-{index}"
        )

    return chunks
