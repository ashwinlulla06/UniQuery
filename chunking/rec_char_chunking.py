from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
from pathlib import Path
from langchain_core.documents import Document
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

def load_cached_documents():
    CACHE_PATH = Path(
        'document-cache/extracted_documents.jsonl'
    )

    documents = []

    with CACHE_PATH.open("r", encoding= 'utf-8') as cache_file:
        for line in cache_file:
            record = json.loads(line)

            doc = Document(
                page_content=record['page_content'],
                metadata = record['metadata']
            )

            documents.append(doc)

    return documents

def rec_char_chunking(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    documents = load_cached_documents()

    if not documents:
        print('No documents found!')
        return

    chunks = text_splitter.split_documents(documents)

    print("Number of Chunks: ", len(chunks))

    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"{chunk.metadata['file_name']}__chunk-{i+1}"

    return chunks
