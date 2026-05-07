import streamlit as st
from rag_pipeline import process_documents, chat_stream, invalidate_db_cache, CHROMA_CHAT
from chroma_manager import reset_chat_db
import gc
import time

st.set_page_config(page_title="Chat", page_icon="💬")

st.markdown(
    "<style>[data-testid='stSidebarNav'] { display: none; }</style>",
    unsafe_allow_html=True,
)

gc.collect()

# ---------------------------
# Chat
# ---------------------------
st.subheader("💬 Chat with your documents")

if "all_chats" not in st.session_state:
    st.session_state.all_chats = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = []

if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None

col1, col2 = st.columns([4, 1])

placeholder = st.empty()

with col1:
    query = st.text_input("Ask a question", label_visibility="collapsed")

with col2:
    new_chat = st.button("➕")
    if new_chat:
     if st.session_state.current_chat:
        st.session_state.all_chats.append(st.session_state.current_chat)
        st.session_state.current_chat = []
        st.session_state.selected_chat = None
        st.rerun()


# Selected conversation
if st.session_state.selected_chat is not None:
    chat = st.session_state.all_chats[st.session_state.selected_chat]
else:
    chat = st.session_state.current_chat

for item in chat:
    st.markdown(f"🧑 **You:** {item['question']}")
    st.markdown(f"🤖 **AI:** {item['answer']}")
    st.markdown("---")


if st.button("Ask", type="primary"):
    if query:
        if (
            not st.session_state.current_chat
            or st.session_state.current_chat[-1]["question"] != query
        ):
            full_response = ""

            for chunk in chat_stream(query):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
                time.sleep(0.02)

            placeholder.markdown(full_response)

            st.session_state.current_chat.append({
                "question": query,
                "answer": full_response
            })
        else:
            st.warning("You already asked this question.")

with st.sidebar:
    if st.button("📚 Generate Handbook"):
        st.switch_page("pages/handbook.py")
    
    st.divider()

    st.header("📂 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if "last_uploaded_signature" not in st.session_state:
        st.session_state.last_uploaded_signature = None

    if uploaded_files:
        current_signature = tuple(
            (file.name, file.size) for file in uploaded_files
        )
        if current_signature != st.session_state.last_uploaded_signature:
            with st.spinner("Processing documents..."):
                process_documents(uploaded_files, chroma_path=CHROMA_CHAT)
            st.session_state.last_uploaded_signature = current_signature
            st.success("Documents processed!")
        else:
            st.info("Documents already processed ✅")

        for f in uploaded_files:
            st.write(f"📄 {f.name}")
    else:
        st.warning("Please upload files first.")

    st.divider()

    st.subheader("🕘 Chat History")

    if st.session_state.all_chats:
      for i, chat in enumerate(st.session_state.all_chats):
        title = chat[0]["question"] if chat else f"Chat {i+1}"

        if st.button(title, key=f"chat_{i}"):
            st.session_state.selected_chat = i
    else:
      st.info("No questions asked yet.")
    
    st.divider()
    
    if st.button("♻️ Reset Knowledge Base"):
     reset_chat_db()
     invalidate_db_cache()
    
