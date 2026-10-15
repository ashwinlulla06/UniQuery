import torch
from sentence_transformers import CrossEncoder

from config import (
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MODEL,
)


def choose_device():
    if RERANKER_DEVICE != "auto":
        return RERANKER_DEVICE

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


DEVICE = choose_device()
BATCH_SIZE = (
    RERANKER_BATCH_SIZE
    if RERANKER_BATCH_SIZE > 0
    else (8 if DEVICE == "cuda" else 1)
)

model = CrossEncoder(
    RERANKER_MODEL,
    device=DEVICE,
)


def rerank_documents(question, retrieved_docs):
    if not retrieved_docs:
        return []

    pairs = [
        (question, document.page_content)
        for document in retrieved_docs
    ]
    scores = model.predict(
        pairs,
        batch_size=BATCH_SIZE,
    )

    ranked_documents = sorted(
        zip(scores, retrieved_docs),
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        document
        for _, document in ranked_documents[:5]
    ]
