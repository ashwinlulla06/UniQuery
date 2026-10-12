# Local GGUF embeddings

Start the local embedding server and keep its PowerShell window open:

```powershell
cd D:\UniQuery-runtime\llama-cpp
.\llama-server.exe `
  -m "D:\huggingface_cache\gguf\Qwen3-Embedding-4B-Q4_K_M.gguf" `
  --embedding --pooling last `
  --host 127.0.0.1 --port 8081 --no-webui `
  -ngl 99 -c 2048 -b 2048 -ub 2048
```

`embeddings.py` sends text to that server and receives an embedding. It does
not download or load a Hugging Face model.

The GGUF vector databases use explicit names ending in `_gguf_q4km`. The old
vector databases therefore remain separate and unchanged.

Create the recursive-character GGUF database with:

```powershell
python -c "from chunking.rec_char_chunking import rec_char_chunking; from vector_stores.storing_vectors import storing_vectors; storing_vectors(rec_char_chunking(), 'vector_stores/rec_char_vectordb_gguf_q4km')"
```

Then run the application:

```powershell
python app.py
```
