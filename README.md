# UniQuery

UniQuery is an advanced retrieval-augmented generation (RAG) assistant for
university policies, notices, regulations, and student-service documents. It
processes native and scanned documents, combines semantic and keyword
retrieval, reranks candidates, and produces grounded answers with source
context.

The Northbridge Institute of Technology documents in `knowledge-base/` are
synthetic and are intended only for academic development, demonstration, and
evaluation.

## Pipeline

```text
University documents
        |
        v
Text extraction / PaddleOCR
        |
        v
Extracted-document cache
        |
        v
Chunking -> GGUF embeddings -> Chroma
                                |
Student query ------------------+
        |                       |
        +-> BM25 retrieval      +-> Vector retrieval
                    \           /
                     RRF fusion
                         |
                    Qwen reranker
                         |
               Gemini grounded answer
                         |
                     Gradio UI
```

## Main features

- Native PDFs, scanned PDFs, TXT, Markdown, PNG, JPG, and JPEG
- PaddleOCR text extraction with confidence scores and bounding boxes
- Cached extracted pages for repeatable chunking experiments
- Recursive, page-level semantic, and file-level semantic chunking
- Qwen GGUF embeddings served locally through llama.cpp
- Chroma vector storage and BM25 keyword retrieval
- Reciprocal Rank Fusion and Qwen reranking
- Selective query rewriting when the local TinyLlama model is cached
- Grounded Gemini answers and a Gradio interface
- MRR, nDCG, Recall@K, Hit Rate@K, and Precision@1 evaluation
- Optional Qwen-judged answer evaluation (accuracy, completeness, relevance)

## Project structure

```text
UniQuery/
|-- app.py
|-- config.py
|-- embeddings.py
|-- storing_documents.py
|-- document_processing/
|-- chunking/
|-- vector_stores/
|-- retrieving_and_answering/
|-- reranking/
|-- rag_evals/
|-- knowledge-base/
|-- proof_of_ocr/
|-- .env.example
|-- requirements.txt
`-- requirements-cpu.txt
```

## Setup

Requirements:

- Python 3.12
- llama.cpp with `llama-server`
- A Qwen embedding model in GGUF format
- A Google AI API key
- Access to this private repository

### 1. Clone and create an environment

Windows PowerShell:

```powershell
git clone https://github.com/ashwinlulla06/UniQuery---Private.git
cd UniQuery---Private
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Linux or macOS:

```bash
git clone https://github.com/ashwinlulla06/UniQuery---Private.git
cd UniQuery---Private
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
cp .env.example .env
```

`requirements.txt` represents the project's Windows CUDA environment.
`requirements-cpu.txt` is the CPU-oriented alternative. If a pinned framework
wheel is unavailable for a particular OS, install its compatible official
wheel first.

### 2. Configure local settings

Edit `.env`:

```env
GOOGLE_API_KEY=your_google_ai_api_key
EMBEDDING_SERVER_URL=http://127.0.0.1:8081
RERANKER_DEVICE=auto
```

The application resolves relative paths from the repository root, so it can be
launched from another working directory. Machine-specific paths and secrets
belong in `.env`, which Git ignores.

### 3. Start the embedding server

The server can be launched from any directory:

```powershell
llama-server `
  -hf "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M" `
  --embedding `
  --pooling last `
  --host 127.0.0.1 `
  --port 8081 `
  --no-webui
```

For CPU-only execution, add `--device none --threads 6`. See
[EMBEDDINGS.md](EMBEDDINGS.md) for Linux/macOS syntax and configuration.

### 4. Extract documents

```powershell
python -m storing_documents
```

This writes `document-cache/extracted_documents.jsonl`. The cache avoids
repeating OCR for every chunking experiment.

### 5. Build a vector store

Example recursive-character experiment:

```powershell
python -c "from chunking.rec_char_chunking import rec_char_chunking; from vector_stores.storing_vectors import storing_vectors; storing_vectors(rec_char_chunking(400, 75), 'vector_stores/rec_char_400')"
```

Example page-level semantic experiment:

```powershell
python -c "from chunking.page_level_semantic_chunking import page_level_chunking; from vector_stores.storing_vectors import storing_vectors; storing_vectors(page_level_chunking(), 'vector_stores/page_level_semantic')"
```

Vector databases are generated locally and are not committed. Rebuild them
whenever the embedding model changes.

### 6. Select the matching experiment

Set these values in `.env`:

```env
DB_NAME=vector_stores/page_level_semantic
QUESTION_SET=rag_evals/questions/test_page_level_semantic_percentile_95.jsonl
```

The vector store and golden dataset must use the same chunking method and
settings because chunk IDs depend on those choices.

### 7. Run

```powershell
python app.py
```

Evaluation:

```powershell
python -m rag_evals.evaluator
```

Answer evaluation requires `QWEN_API_KEY` (a Groq API key) in `.env`. The
query rewriter uses a locally cached TinyLlama model and will continue with
the original query if those weights are not present. It does not download the
model automatically. The embedding model and `llama-server` still need to be
installed and started separately before running the assistant.

## Portable configuration

The following environment variables may be overridden without editing source
code:

- `KNOWLEDGE_BASE_DIR`
- `DOCUMENT_CACHE_PATH`
- `PROOF_OF_OCR_DIR`
- `DB_NAME`
- `QUESTION_SET`
- `EMBEDDING_SERVER_URL`
- `EMBEDDING_REQUEST_TIMEOUT`
- `RERANKER_MODEL`
- `RERANKER_DEVICE`
- `RERANKER_BATCH_SIZE`
- `HF_HOME`
- `QUERY_REWRITE_MODEL`

`RERANKER_DEVICE=auto` selects CUDA when available, Apple MPS on a compatible
Mac, and otherwise the CPU.

## Evaluation datasets

The repository includes golden datasets for:

- Recursive character splitting: 1000/200, 500/100, 400/75, and 300/50
- Page-level semantic splitting: percentile threshold 95
- File-level semantic splitting: percentile threshold 95

Unanswerable questions remain in the datasets for future grounded-refusal
testing, while the current retrieval dashboard evaluates answerable questions.

## Limitations

- OCR quality depends on scan resolution, orientation, and noise.
- Weak OCR is flagged, but a vision-language fallback is not yet automatic.
- The embedding server and Gemini API must be available while the app runs.
- The synthetic knowledge base is intentionally small.
- Changing chunking or embeddings can invalidate stored chunk IDs and vectors.