from functools import lru_cache
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from pdf_processor import extract_text_from_pdf

CHROMA_PATH = "chroma_chat"

# ─────────────────────────────────────────────────────────────────────────────
# SINGLETONS  (created once, reused forever — like moving work outside a loop)
#
# BEFORE: get_embedding_function() built a new OllamaEmbeddings object on
#         every call → O(init_cost) × number_of_calls
# AFTER:  @lru_cache ensures it is built exactly once → O(init_cost) × 1
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_model() -> ChatOllama:
    return ChatOllama(model="phi3")


@lru_cache(maxsize=1)                   # ← THE KEY FIX (was missing before)
def get_embedding_function() -> OllamaEmbeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


@lru_cache(maxsize=1)                   # ← NEW: splitter config never changes
def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )


@lru_cache(maxsize=1)                   # ← NEW: one persistent DB connection
@st.cache_resource
def get_chat_db() -> Chroma:
    """
    Open the Chroma vector store once and reuse the connection.

    BEFORE: Chroma(...) was instantiated inside add_to_chroma() AND
            chat_stream() — a fresh (expensive) connection on every call.
    AFTER:  one connection, shared across all operations → O(1) overhead.
    """
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Call this after reset_database() so the cached DB handle is discarded
# ─────────────────────────────────────────────────────────────────────────────
def invalidate_db_cache() -> None:
    get_chat_db.cache_clear()


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def split_documents(documents):
    # get_text_splitter() is now O(1) — returns the cached instance
    return get_text_splitter().split_documents(documents)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk ID generation  (algorithm unchanged — already O(n), which is optimal)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page   = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Store in Chroma
# ─────────────────────────────────────────────────────────────────────────────

def add_to_chroma(chunks):
    db = get_chat_db()                # O(1) — reuses cached connection

    existing_items = db.get(include=[])
    existing_ids   = set(existing_items["ids"])  # set → O(1) lookups below

    chunks = calculate_chunk_ids(chunks)

    # List comprehension with set lookup: O(n) total, not O(n²)
    new_chunks = [
        chunk for chunk in chunks
        if chunk.metadata["id"] not in existing_ids
    ]

    if new_chunks:
        new_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Process uploaded PDFs
# ─────────────────────────────────────────────────────────────────────────────

def process_documents(uploaded_files):
    all_docs = []

    for file in uploaded_files:
        docs = extract_text_from_pdf(file)
        all_docs.extend(docs)

    chunks = split_documents(all_docs)
    add_to_chroma(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Chat (RAG query)
# ─────────────────────────────────────────────────────────────────────────────

def chat_stream(query: str):
    db = get_chat_db()                # O(1) — reuses cached connection
                                        # BEFORE: new Chroma(...) every query

    results = db.similarity_search_with_score(query, k=5)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _ in results]
    )

    prompt = f"""
Answer the question based only on the following context:

{context_text}

---

Question: {query}
Answer:
"""

    model = get_model()                 # O(1) — cached

    for chunk in model.stream(prompt):
        yield chunk.content
