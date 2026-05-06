import streamlit as st

st.markdown(
    "<style>[data-testid='stSidebarNav'] { display: none; }</style>",
    unsafe_allow_html=True,
)


st.title("🤖 AI Assistant")

st.markdown("""
Welcome! Choose what you want to do:
""")

if st.button("💬 Go to Chat"):
    st.switch_page("pages/chat.py")

if st.button("📚 Generate Handbook"):
    st.switch_page("pages/handbook.py")