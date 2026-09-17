from langchain_experimental.text_splitter import SemanticChunker
import os
from pathlib import Path
import json
from pathlib import Path
from langchain_core.documents import Document
from embeddings import get_embeddings

os.environ['HF_HOME'] = 'D:/huggingface_cache'

CACHE_PATH = Path(
    "document-cache/extracted_documents.jsonl"
)


def load_cached_documents():
    documents = []

    with CACHE_PATH.open(
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
    
    embedding_model = get_embeddings()

    text_splitter = SemanticChunker(
        embeddings= embedding_model,
        breakpoint_threshold_type = "percentile",
        breakpoint_threshold_amount = 95,
    )

    documents = load_cached_documents()

    if not documents:
        print("No documents were extracted.")
        return

    chunks = text_splitter.split_documents(documents)

    print('Number of Chunks: ', len(chunks))

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"{chunk.metadata['file_name']}__chunk-{i+1}"

    return chunks
