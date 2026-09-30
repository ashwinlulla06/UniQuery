import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


KNOWLEDGE_BASE_DIR = resolve_project_path(
    os.getenv("KNOWLEDGE_BASE_DIR", "knowledge-base")
)
DOCUMENT_CACHE_PATH = resolve_project_path(
    os.getenv(
        "DOCUMENT_CACHE_PATH",
        "document-cache/extracted_documents.jsonl",
    )
)
PROOF_OF_OCR_DIR = resolve_project_path(
    os.getenv("PROOF_OF_OCR_DIR", "proof_of_ocr")
)

DB_NAME = str(
    resolve_project_path(
        os.getenv(
            "DB_NAME",
            "vector_stores/page_level_semantic",
        )
    )
)
QUESTION_SET = resolve_project_path(
    os.getenv(
        "QUESTION_SET",
        "rag_evals/questions/"
        "test_page_level_semantic_percentile_95.jsonl",
    )
)

EMBEDDING_SERVER_URL = os.getenv(
    "EMBEDDING_SERVER_URL",
    "http://127.0.0.1:8081",
).rstrip("/")
EMBEDDING_REQUEST_TIMEOUT = int(
    os.getenv("EMBEDDING_REQUEST_TIMEOUT", "120")
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "Qwen/Qwen3-Reranker-0.6B",
)
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
RERANKER_BATCH_SIZE = int(
    os.getenv("RERANKER_BATCH_SIZE", "0")
)
