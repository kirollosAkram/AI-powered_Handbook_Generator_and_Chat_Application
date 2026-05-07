import streamlit as st


def _clear_collection(chroma_path: str) -> None:
    """
    Clear all documents from the collection without deleting files on disk.
    This avoids Windows file-locking errors entirely.
    """
    try:
        from rag_pipeline import get_db
        db = get_db(chroma_path)
        db.delete_collection()
    except Exception:
        pass
    finally:
        try:
            from rag_pipeline import get_db
            get_db.cache_clear()
        except Exception:
            pass
        st.cache_resource.clear()


def reset_chat_db() -> None:
    _clear_collection("chroma_chat")


def reset_handbook_db() -> None:
    _clear_collection("chroma_handbook")