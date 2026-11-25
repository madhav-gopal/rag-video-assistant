import os
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR       = "vector_db"
COLLECTION_NAME  = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def _clear_vector_store():
    """
    Wipe the persisted ChromaDB directory so the previous video's
    embeddings can never bleed into the current session.

    On Windows, ChromaDB holds open file handles on its .bin files while
    the client is alive. We must delete the collection via ChromaDB's own
    API first (which closes the SQLite/MMAP handles), then remove the folder.
    """
    if os.path.exists(CHROMA_DIR):
        print(f"[vector_store] Clearing old vector DB at '{CHROMA_DIR}' …")

        # Step 1 — ask ChromaDB to close its own file handles
        try:
            import chromadb
            import gc
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass          # collection may not exist yet — that's fine
            del client        # drop the Python reference
            gc.collect()      # encourage CPython to close the underlying handles
        except Exception as e:
            print(f"[vector_store] Warning while closing ChromaDB client: {e}")

        # Step 2 — now the files are unlocked; delete the directory
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    os.makedirs(CHROMA_DIR, exist_ok=True)


def build_vector_store(transcript: str) -> Chroma:
    """
    Build a FRESH vector store for the current video's transcript.
    Always clears any previously stored data first.
    """
    print("[vector_store] Building vector store …")

    # ✅ Wipe old data — guarantees only the current video is in the DB
    _clear_vector_store()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(f"[vector_store] Stored {len(docs)} chunks for current video.")
    return vector_store


def load_vector_store() -> Chroma:
    """Load the already-built vector store (current video only)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
