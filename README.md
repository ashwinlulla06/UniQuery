# UniQuery

UniQuery is an advanced retrieval-augmented generation (RAG) assistant for university policies, notices, regulations, and student-service documents. It accepts several document formats, retrieves supporting passages through both semantic and keyword search, reranks them, and generates a grounded answer with source context.

This repository is an academic project. The Northbridge Institute of Technology documents in `knowledge-base/` are synthetic and are intended only for development, demonstration, and evaluation. They are not official policies of a real institution.

## What the project demonstrates

- Processing native-text PDFs, scanned PDFs, TXT, Markdown, PNG, JPG, and JPEG files
- Extracting scanned text with PaddleOCR and recording confidence scores and bounding boxes
- Caching extracted pages so OCR does not run during every experiment
- Comparing recursive character chunking with page-level and file-level semantic chunking
- Generating embeddings locally with a quantized Qwen GGUF model served by llama.cpp
- Persisting embeddings and metadata in Chroma
- Combining BM25 keyword retrieval and vector retrieval
- Merging ranked results with Reciprocal Rank Fusion (RRF)
- Reranking candidates with `Qwen/Qwen3-Reranker-0.6B`
- Producing grounded answers with Gemini and displaying the supporting context
- Evaluating retrieval with MRR, nDCG, Recall@K, Hit Rate@K, and Precision@1

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

## Knowledge base

The project contains 30 synthetic university documents grouped into ten areas:

```text
academic/       attendance/     career/          courses/
examinations/   fees/           hostel/          library/
scholarships/   student-services/
```

The collection deliberately mixes native documents and scanned material so that both ordinary text extraction and OCR can be tested.

## Project structure

```text
UniQuery/
|-- app.py                         Gradio assistant
|-- config.py                      Active vector store and question set
|-- embeddings.py                  llama.cpp embedding client
|-- storing_documents.py           Document extraction and cache creation
|-- document_processing/           PDF, image, text, and Markdown processing
|-- chunking/                      Recursive and semantic chunkers
|-- vector_stores/                 Chroma creation and persisted stores
|-- retrieving_and_answering/      BM25, vector search, RRF, and answer generation
|-- reranking/                     Cross-encoder reranking
|-- rag_evals/                     Golden questions and evaluation dashboard
|-- knowledge-base/                Synthetic university source documents
|-- proof_of_ocr/                  OCR visualizations
|-- visualizations/                Embedding and chunking plots
`-- requirements.txt
```

## Requirements

- Python 3.12
- Windows PowerShell commands are shown below
- llama.cpp with `llama-server`
- A Qwen embedding model in GGUF format
- A Google AI API key for answer generation
- An NVIDIA GPU is recommended for the Qwen reranker; PaddleOCR is configured to use the CPU

The pinned environment uses the CUDA-enabled PyTorch build listed in `requirements.txt`. If that build is not supported by your machine, install the appropriate PyTorch version for your CUDA setup before installing the remaining packages.

## Installation

```powershell
git clone https://github.com/ashwinlulla06/UniQuery.git
cd UniQuery

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_ai_api_key
```

Do not commit the `.env` file or your API key.

## Start the local embedding server

The embedding client in `embeddings.py` expects an OpenAI-compatible llama.cpp server at `http://127.0.0.1:8081`.

```powershell
<path-to-llama-server.exe> -m "<path-to-qwen-embedding-model.gguf>" --embedding --host 127.0.0.1 --port 8081
```

Keep this terminal open while creating vector stores, running evaluations, or using the assistant. Model-specific llama.cpp builds may also require a pooling option; use the option recommended by the selected Qwen GGUF model card.

## Prepare the documents

Run extraction once to process the knowledge base and create `document-cache/extracted_documents.jsonl`:

```powershell
python -m storing_documents
```

Later chunking experiments read this cache directly, which avoids repeating CPU OCR for every run.

To build the current page-level semantic vector store:

```powershell
python -c "from chunking.page_level_semantic_chunking import page_level_chunking; from vector_stores.storing_vectors import storing_vectors; storing_vectors(page_level_chunking(), 'vector_stores/page_level_semantic')"
```

Other available strategies can be created with the same storage function:

```python
from chunking.rec_char_chunking import rec_char_chunking
from chunking.file_level_semantic_chunking import file_level_chunking
from vector_stores.storing_vectors import storing_vectors

storing_vectors(rec_char_chunking(400, 75), "vector_stores/rec_char_400")
storing_vectors(file_level_chunking(), "vector_stores/file_level_semantic")
```

Building a store replaces the existing Chroma collection at that destination. Keep the embedding model unchanged when comparing stores because chunk IDs and vectors are tied to the selected configuration.

## Select an experiment

`config.py` selects the vector store used by retrieval and the matching JSONL question set used by evaluation:

```python
DB_NAME = "vector_stores/page_level_semantic"
QUESTION_SET = "rag_evals/questions/test_page_level_semantic_percentile_95.jsonl"
```

Always pair a question set with the vector store built from the same chunking method and settings. The repository includes question sets for:

- Recursive character splitting: 1000/200, 500/100, 400/75, and 300/50 chunk size/overlap
- Page-level semantic splitting: percentile threshold 95
- File-level semantic splitting: percentile threshold 95

## Run the assistant

Start the embedding server first, then run:

```powershell
python app.py
```

Open the local Gradio address printed in the terminal. A query is sent through vector retrieval and BM25, fused with RRF, reranked, and then passed to Gemini with the retrieved context. The interface displays both the answer and its supporting passages.

## Run retrieval evaluation

```powershell
python -m rag_evals.evaluator
```

The evaluation dashboard uses the question set selected in `config.py`. It currently evaluates answerable questions and reports:

- **MRR:** how early the first relevant chunk appears
- **nDCG:** the quality of the overall ranking
- **Recall@K:** how many labelled relevant chunks were retrieved
- **Hit Rate@K:** whether at least one relevant chunk was retrieved
- **Precision@1:** whether the top-ranked chunk was relevant

The JSONL files also retain unanswerable questions for future testing of grounded refusal and no-answer behaviour.

## Main design choices

**Hybrid retrieval.** Vector search handles semantic similarity, while BM25 helps with exact terms such as notice numbers, dates, course codes, and policy names.

**RRF and reranking.** Reciprocal Rank Fusion combines the two retrieval rankings without requiring their raw scores to be directly comparable. A cross-encoder then scores the fused candidates against the query and returns the five strongest passages.

**Grounded generation.** Gemini receives the retrieved passages and conversation history. The system prompt instructs it to answer from the supplied context and acknowledge when the documents do not resolve the question.

**Reproducible evaluation.** Each golden question stores the relevant chunk IDs. Separate datasets are maintained because changing the chunking strategy changes those IDs.

## Current limitations

- OCR quality still depends on scan resolution, orientation, and document noise.
- The current OCR confidence checks flag weak extraction but do not automatically invoke a vision-language fallback.
- The embedding server and Gemini service must both be available while the assistant runs.
- The synthetic knowledge base is intentionally small and does not represent a production university corpus.
- Retrieval labels generated for one chunking configuration cannot be reused unchanged for another.

## Repository

Suggested repository name: **`university-advanced-rag-assistant`**

Current project name: **UniQuery**
