# UniQuery: CPU setup

This guide runs UniQuery without CUDA. PaddleOCR, Chroma, BM25, and the
reranker use the CPU; llama.cpp generates embeddings; Gemini generates the
final answer through its API.

## 1. Create the environment

### Windows PowerShell

```powershell
git clone https://github.com/ashwinlulla06/UniQuery---Private.git
cd UniQuery---Private
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
Copy-Item .env.example .env
```

### Linux or macOS

```bash
git clone https://github.com/ashwinlulla06/UniQuery---Private.git
cd UniQuery---Private
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cpu.txt
cp .env.example .env
```

If a pinned PyTorch or PaddlePaddle wheel is unavailable for the operating
system, install the compatible CPU wheel from that framework's official
instructions and then install the remaining requirements.

## 2. Configure the application

Open `.env` and add the Gemini key:

```env
GOOGLE_API_KEY=your_google_ai_api_key
RERANKER_DEVICE=cpu
RERANKER_BATCH_SIZE=1
```

No Python file needs to be edited. Relative paths in `.env` are resolved from
the repository root.

## 3. Start the CPU embedding server

The command may be run from any directory:

```powershell
llama-server `
  -hf "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M" `
  --embedding `
  --pooling last `
  --host 127.0.0.1 `
  --port 8081 `
  --device none `
  --threads 6 `
  --no-webui
```

On Linux or macOS, replace PowerShell backticks with backslashes. Keep the
server terminal open.

A 4B model can be slow or memory-heavy on smaller laptops. A smaller Qwen GGUF
embedding model can be used, but all Chroma databases must then be rebuilt.

## 4. Prepare and run

From the repository root:

```powershell
python -m storing_documents
python -c "from chunking.rec_char_chunking import rec_char_chunking; from vector_stores.storing_vectors import storing_vectors; storing_vectors(rec_char_chunking(400, 75), 'vector_stores/rec_char_400')"
python app.py
```

To evaluate retrieval:

```powershell
python -m rag_evals.evaluator
```

Ensure that `DB_NAME` and `QUESTION_SET` in `.env` describe matching
chunking configurations.

## CPU performance tips

- Reuse `document-cache/extracted_documents.jsonl` so OCR does not repeat.
- Build vector databases only when documents, chunking, or embeddings change.
- Keep the reranker batch size at 1.
- Use a smaller quantized embedding model when memory is limited.
