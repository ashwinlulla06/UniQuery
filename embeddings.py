import requests
from langchain_core.embeddings import Embeddings

from config import EMBEDDING_REQUEST_TIMEOUT, EMBEDDING_SERVER_URL


QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)


class GGUFEmbeddings(Embeddings):

    def __init__(self):
        self.session = requests.Session()
        self.session.trust_env = False

    def embed_text(self, text):
        response = self.session.post(
            f"{EMBEDDING_SERVER_URL}/v1/embeddings",
            json={"input": text.replace("\n", " ")},
            timeout=EMBEDDING_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_documents(self, texts):
        return [self.embed_text(text) for text in texts]

    def embed_query(self, text):
        return self.embed_text(QUERY_INSTRUCTION + text)


def get_embeddings():
    return GGUFEmbeddings()
