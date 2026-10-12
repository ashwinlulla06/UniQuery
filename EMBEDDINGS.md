# Local GGUF embeddings

UniQuery sends embedding requests to an OpenAI-compatible llama.cpp server. The
server can be started from any directory; it does not need to be inside this
repository.

## Download and start automatically from Hugging Face

The following command downloads the selected GGUF file into the normal
Hugging Face cache and starts the server.

### Windows PowerShell

```powershell
llama-server `
  -hf "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M" `
  --embedding `
  --pooling last `
  --host 127.0.0.1 `
  --port 8081 `
  --no-webui
```

Add `--device none --threads 6` for CPU-only execution, or
`--n-gpu-layers 99` when the installed llama.cpp build supports the machine's
GPU.

### Linux and macOS

```bash
llama-server \
  -hf "Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M" \
  --embedding \
  --pooling last \
  --host 127.0.0.1 \
  --port 8081 \
  --no-webui
```

For CPU-only execution, add `--device none --threads 6`.

## Configuration

Copy `.env.example` to `.env`. The default client setting is:

```env
EMBEDDING_SERVER_URL=http://127.0.0.1:8081
```

If the server runs on another host or port, change only that environment
variable. No Python source file needs to be edited.

To use a custom cache location, set `HF_HOME` in the shell that starts
`llama-server`. Do not put a personal cache path in committed Python code.

`embeddings.py` only calls the server. It does not load the GGUF model into
Python.

## Build a vector database

Start the server first, then run this command from the repository root:

```powershell
python -c "from chunking.rec_char_chunking import rec_char_chunking; from vector_stores.storing_vectors import storing_vectors; storing_vectors(rec_char_chunking(400, 75), 'vector_stores/rec_char_400')"
```

A vector database must be rebuilt if the embedding model changes. Do not query
a database made with a different embedding model.
