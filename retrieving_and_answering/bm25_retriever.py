import re
from langchain_community.retrievers import BM25Retriever
from vector_stores.storing_vectors import COLLECTION_NAME
from config import DB_NAME
from langchain_chroma import Chroma
from embeddings import get_embeddings
from langchain_core.documents import Document

def tokenize(text):
    return (
        re.findall(
            r"[a-z0-9]+(?:[./-][a-z0-9]+)*",
            text.lower(),
        )
    )

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_NAME,
    embedding_function=get_embeddings(),
)
    
if vector_store._collection.count() == 0:
    raise ValueError(
        f"Vector store '{DB_NAME}' is empty. Create it before running retrieval."
    )

stored_data = vector_store.get(include=['documents', 'metadatas'])

documents = [
    Document(
        page_content = doc,
        metadata = metadata
    )
    for doc, metadata in zip(stored_data['documents'], stored_data['metadatas'])
]

bm25_retriever = BM25Retriever.from_documents(
    documents = documents,
    preprocess_func=tokenize
)

bm25_retriever.k = 15

def keyword_retriever(question):

    return bm25_retriever.invoke(question)
