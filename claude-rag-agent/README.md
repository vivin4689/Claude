# Local-file RAG chatbot

Chat with documents on your own disk. Point the app at an absolute directory
path, it indexes the files, and you ask questions about them.

| Piece       | Choice                                                        | Cost |
|-------------|---------------------------------------------------------------|------|
| UI          | Streamlit                                                     | free |
| Embeddings  | HuggingFace `all-MiniLM-L6-v2`, run locally on CPU           | free |
| Vector DB   | ChromaDB, in-memory (nothing written to disk)                 | free |
| LLM         | Groq Cloud `llama-3.3-70b-versatile` via `ChatGroq`          | free tier |

## Setup

```bash
cd claude-rag-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then paste your key from https://console.groq.com/keys
streamlit run app.py
```

The first run downloads the embedding model (~90 MB) into the HuggingFace cache.

## Usage

1. In the sidebar, enter an **absolute** directory path, e.g. `/home/me/notes`
   or `C:\Users\me\notes`. `~` is expanded; relative paths are rejected.
2. Choose file types and whether to include subdirectories.
3. Click **Scan & index**.
4. Ask questions in the chat box. Each answer lists the chunks it used under
   **Sources**.

Supported file types: PDF, DOCX, and plain text (`.txt`, `.md`, `.csv`,
`.json`, code files, and so on). Hidden folders, `.git`, `node_modules`,
virtualenvs, and files over 20 MB are skipped.

The index lives in memory only, so it is gone when the Streamlit process
stops. Re-indexing or clicking **Clear** replaces it.
