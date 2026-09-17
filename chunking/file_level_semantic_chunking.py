from langchain_experimental.text_splitter import SemanticChunker
import os
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import json
from pathlib import Path
from langchain_core.documents import Document
from collections import defaultdict
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

def merging_pages_of_same_file():

    documents = load_cached_documents()

    if not documents:
        return []
    
    merged_document = []

    documents_by_file = defaultdict(list) 

    for document in documents:
        source = str(document.metadata.get('source', ''))
        documents_by_file[source].append(document)

    for source, documents in documents_by_file.items():

        metadata = {}

        documents.sort(
            key = lambda page: page.metadata.get('page_number', 0)
        )

        merged_text = "\n\n".join(page.page_content for page in documents)

        metadata['source'] = str(source)
        metadata['file_name'] = os.path.basename(source)
        metadata['file_type'] = os.path.splitext(source)[1].lower().lstrip('.')
        metadata['document_id'] = os.path.basename(source).split('-')[0]
        metadata['folder'] = os.path.dirname(source).split('\\')[-1]
        
        merged_document.append(
            Document(
                page_content=merged_text,
                metadata=metadata
            )
        )

    return merged_document


def file_level_chunking():
    
    embedding_model = get_embeddings()

    text_splitter = SemanticChunker(
        embeddings= embedding_model,
        breakpoint_threshold_type = "percentile",
        breakpoint_threshold_amount = 95,
    )

    documents = merging_pages_of_same_file()

    if not documents:
        print("No documents were extracted.")
        return

    chunks = text_splitter.split_documents(documents)

    print('Number of Chunks: ', len(chunks))

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"{chunk.metadata['file_name']}__chunk-{i+1}"
    
    return chunks    
