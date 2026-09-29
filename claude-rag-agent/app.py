"""Local-file RAG chatbot.

Scans an absolute local directory for documents, embeds them locally with
HuggingFace `all-MiniLM-L6-v2`, stores the chunks in an in-memory ChromaDB
collection, and answers questions with Groq's `llama-3.3-70b-versatile`.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".html", ".css", ".java", ".go", ".rs", ".c",
    ".cpp", ".h", ".sh", ".toml", ".ini", ".log",
}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "chroma_db"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # skip files larger than 20 MB

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the \
user's local documents. Use ONLY the context below to answer. If the answer \
is not in the context, say you don't know. Cite the source file names you \
used in square brackets, e.g. [notes.md].

Context:
{context}"""


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading local embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_llm(api_key: str, temperature: float) -> ChatGroq:
    return ChatGroq(model=LLM_MODEL, api_key=api_key, temperature=temperature)


# ---------------------------------------------------------------------------
# Directory parsing
# ---------------------------------------------------------------------------

def validate_directory(raw_path: str) -> tuple[Path | None, str | None]:
    """Return (resolved_path, error_message)."""
    raw_path = raw_path.strip().strip('"').strip("'")
    if not raw_path:
        return None, "Enter a directory path."
    path = Path(os.path.expanduser(raw_path))
    if not path.is_absolute():
        return None, f"`{raw_path}` is not an absolute path."
    if not path.exists():
        return None, f"`{path}` does not exist."
    if not path.is_dir():
        return None, f"`{path}` is not a directory."
    if not os.access(path, os.R_OK):
        return None, f"`{path}` is not readable."
    return path.resolve(), None


def scan_directory(root: Path, recursive: bool, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")
            ]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() in extensions:
                    files.append(p)
    else:
        files = [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        ]
    return sorted(
        p for p in files if p.stat().st_size <= MAX_FILE_BYTES
    )


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    else:
        loader = TextLoader(str(path), autodetect_encoding=True)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = str(path)
        d.metadata["filename"] = path.name
    return docs


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_vectorstore(
    files: list[Path], chunk_size: int, chunk_overlap: int
) -> tuple[Chroma | None, int, list[str]]:
    """Load, split and embed files into a fresh in-memory Chroma collection."""
    documents: list[Document] = []
    errors: list[str] = []
    progress = st.progress(0.0, text="Reading files...")
    for i, path in enumerate(files, start=1):
        try:
            documents.extend(load_file(path))
        except Exception as exc:  # noqa: BLE001 - report and continue
            errors.append(f"{path.name}: {exc}")
        progress.progress(i / len(files), text=f"Read {i}/{len(files)}: {path.name}")
    progress.empty()

    documents = [d for d in documents if d.page_content.strip()]
    if not documents:
        return None, 0, errors

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    with st.spinner(f"Embedding {len(chunks)} chunks locally..."):
        # No persist_directory -> ephemeral, in-memory ChromaDB.
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            collection_name=f"rag_{uuid.uuid4().hex}",
            collection_metadata={"hnsw:space": "cosine"},
        )
    return vectorstore, len(chunks), errors


def reset_index() -> None:
    old = st.session_state.get("vectorstore")
    if old is not None:
        try:
            old.delete_collection()
        except Exception:  # noqa: BLE001
            pass
    st.session_state.vectorstore = None
    st.session_state.indexed_files = []
    st.session_state.chunk_count = 0


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def init_state() -> None:
    defaults = {
        "vectorstore": None,
        "indexed_files": [],
        "chunk_count": 0,
        "indexed_root": None,
        "messages": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")

        env_key = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input(
            "Groq API key",
            value=env_key,
            type="password",
            help="Free key at https://console.groq.com/keys. "
                 "Can also be set as GROQ_API_KEY in .env.",
        )

        st.subheader("Documents")
        raw_dir = st.text_input(
            "Absolute directory path",
            placeholder="/home/you/Documents/notes"
            if os.name != "nt" else r"C:\Users\you\Documents\notes",
        )
        recursive = st.checkbox("Include subdirectories", value=True)
        ext_choice = st.multiselect(
            "File types",
            options=sorted(SUPPORTED_EXTENSIONS),
            default=[".md", ".pdf", ".txt", ".docx"],
        )

        with st.expander("Chunking & retrieval"):
            chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
            chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=25)
            top_k = st.slider("Chunks retrieved (k)", 1, 15, 4)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, step=0.05)

        col1, col2 = st.columns(2)
        index_clicked = col1.button("Scan & index", type="primary", use_container_width=True)
        if col2.button("Clear", use_container_width=True):
            reset_index()
            st.session_state.messages = []
            st.rerun()

        if index_clicked:
            path, error = validate_directory(raw_dir)
            if error:
                st.error(error)
            elif not ext_choice:
                st.error("Select at least one file type.")
            elif chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
            else:
                files = scan_directory(path, recursive, set(ext_choice))
                if not files:
                    st.warning(f"No matching files found in `{path}`.")
                else:
                    reset_index()
                    vs, n_chunks, errors = build_vectorstore(
                        files, chunk_size, chunk_overlap
                    )
                    if vs is None:
                        st.error("No readable text found in the matching files.")
                    else:
                        st.session_state.vectorstore = vs
                        st.session_state.indexed_files = [str(f) for f in files]
                        st.session_state.chunk_count = n_chunks
                        st.session_state.indexed_root = str(path)
                        st.session_state.messages = []
                        st.success(f"Indexed {len(files)} files ({n_chunks} chunks).")
                    for err in errors:
                        st.warning(f"Skipped {err}")

        if st.session_state.vectorstore is not None:
            st.caption(
                f"**Index:** {st.session_state.indexed_root}  \n"
                f"{len(st.session_state.indexed_files)} files · "
                f"{st.session_state.chunk_count} chunks (in memory)"
            )
            with st.expander("Indexed files"):
                root = st.session_state.indexed_root
                for f in st.session_state.indexed_files:
                    st.text(os.path.relpath(f, root))

    return {"api_key": api_key, "top_k": top_k, "temperature": temperature}


def format_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        label = d.metadata.get("filename", "unknown")
        if page is not None:
            label += f" (page {page + 1})"
        parts.append(f"[{label}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def to_history(messages: list[dict]) -> list:
    history = []
    for m in messages[-10:]:  # last 5 exchanges keeps the prompt small
        cls = HumanMessage if m["role"] == "user" else AIMessage
        history.append(cls(content=m["content"]))
    return history


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['label']}** — `{s['source']}`")
            st.caption(s["snippet"])


def main() -> None:
    st.set_page_config(page_title="Local RAG Chat", page_icon="📂", layout="wide")
    init_state()
    settings = sidebar()

    st.title("📂 Local-file RAG chatbot")
    st.caption(
        f"Embeddings: `{EMBEDDING_MODEL}` (local) · Vector DB: ChromaDB (in-memory) "
        f"· LLM: Groq `{LLM_MODEL}`"
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))

    ready = st.session_state.vectorstore is not None and bool(settings["api_key"])
    if st.session_state.vectorstore is None:
        st.info("Enter an absolute directory path in the sidebar and click **Scan & index**.")
    elif not settings["api_key"]:
        st.info("Add your Groq API key in the sidebar to start chatting.")

    question = st.chat_input("Ask about your documents...", disabled=not ready)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={"k": settings["top_k"]}
    )
    docs = retriever.invoke(question)
    sources = [
        {
            "label": d.metadata.get("filename", "unknown")
            + (f" (page {d.metadata['page'] + 1})" if "page" in d.metadata else ""),
            "source": d.metadata.get("source", ""),
            "snippet": d.page_content[:300] + ("…" if len(d.page_content) > 300 else ""),
        }
        for d in docs
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | get_llm(settings["api_key"], settings["temperature"]) | StrOutputParser()

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                chain.stream(
                    {
                        "context": format_context(docs),
                        "history": to_history(st.session_state.messages),
                        "question": question,
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface API errors in the UI
            answer = f"⚠️ Groq request failed: {exc}"
            st.error(answer)
        render_sources(sources)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
