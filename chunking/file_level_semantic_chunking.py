import json
from collections import defaultdict
from pathlib import Path

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


def merging_pages_of_same_file():
    documents = load_cached_documents()

    if not documents:
        return []

    documents_by_file = defaultdict(list)

    for document in documents:
        source = str(document.metadata.get("source", ""))
        documents_by_file[source].append(document)

    merged_documents = []

    for source, pages in documents_by_file.items():
        pages.sort(
            key=lambda page: page.metadata.get("page_number", 0)
        )
        merged_text = "\n\n".join(
            page.page_content for page in pages
        )

        # Existing caches may contain Windows separators. Normalizing
        # them keeps the same cache usable on Linux and macOS.
        source_path = Path(source.replace("\\", "/"))
        file_name = source_path.name

        metadata = {
            "source": source_path.as_posix(),
            "file_name": file_name,
            "file_type": source_path.suffix.lower().lstrip("."),
            "document_id": file_name.split("-")[0],
            "folder": source_path.parent.name,
        }

        merged_documents.append(
            Document(
                page_content=merged_text,
                metadata=metadata,
            )
        )

    return merged_documents


def file_level_chunking():
    text_splitter = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
    )

    documents = merging_pages_of_same_file()

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
