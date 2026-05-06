import streamlit as st
from handbook_export import get_history, save_handbook_as_pdf, save_history_entry
from handbook_generator import compile_handbook, generate_handbook_stream, parse_plan_lines
from chroma_manager import reset_handbook_db
from rag_pipeline import process_documents, invalidate_db_cache

st.markdown(
    "<style>[data-testid='stSidebarNav'] { display: none; }</style>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Handbook Generator", page_icon="📚")
st.title("📚 Handbook Generator")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — upload + history
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    if st.button("💬 Chat"):
        st.switch_page("pages/chat.py")
    
    st.divider()
    
    st.subheader("📄 Upload Source Documents")
    uploaded_files = st.file_uploader(
        "Upload files to build the knowledge base",
        accept_multiple_files=True,
        key="uploaded_files",
    )
    if uploaded_files:
        with st.spinner("Processing documents…"):
            process_documents(uploaded_files)
        st.success(f"{len(uploaded_files)} file(s) indexed.")

    st.divider()

    st.subheader("🕘 Handbook History")
    history = get_history()
    if history:
        for item in reversed(history):
            st.markdown(
                f"**{item['topic']}**  \n"
                f"{item['created_at']} · {item['sections']} sections · "
                f"{item['word_count']:,} words"
            )
            st.caption(item["pdf_path"])
    else:
        st.info("No handbooks generated yet.")

# ─────────────────────────────────────────────────────────────────────────────
# Main area — generation
# ─────────────────────────────────────────────────────────────────────────────

# Initializing the key to investigate whether there is a topic or not
if "handbook_topic" not in st.session_state:
    st.session_state.handbook_topic = ""
    
col1, col2 = st.columns([4, 1])

if "selected_handbook" not in st.session_state:
    st.session_state.selected_handbook = None

if "handbook_history" not in st.session_state:
    st.session_state.handbook_history = []


with col1:
    topic = st.text_input(
        "Enter handbook topic",
        placeholder="e.g. Workplace Safety",
        label_visibility="collapsed",
        key="handbook_topic",
    )

with col2:
    new_handbook = st.button("➕")
    if new_handbook:
        try:
            reset_handbook_db()
            invalidate_db_cache()
        except Exception as e:
            st.warning(f"Reset issue: {e}")

        st.session_state.handbook_history = []
        st.session_state.selected_handbook = None
        st.session_state.handbook_topic = ""
        st.rerun()

if st.button("Generate Handbook", type="primary"):
    if not topic:
        st.warning("Please enter a topic first.")   # ← validation moved here
    else:
        reset_handbook_db()

        progress_bar = st.progress(0, text="Planning…")
        preview      = st.empty()

        sections:     list[str]  = []
        writing_plan: str        = ""
        plan_lines:   list[dict] = []
        total:        int        = 0
        
        try:
            for event in generate_handbook_stream(topic):

                if event["type"] == "plan":
                    writing_plan = event["writing_plan"]
                    total        = event["total"]
                    # FIX: plan_lines is now a list[dict] from parse_plan_lines(),
                    #      not a list[str], so compile_handbook receives the right type.
                    plan_lines   = parse_plan_lines(writing_plan)
                    progress_bar.progress(0, text=f"Writing 0 / {total} sections…")

                elif event["type"] == "section_done":
                    sections.append(event["section_text"])
                    pct = int(len(sections) / total * 100)
                    progress_bar.progress(pct, text=f"Writing {len(sections)} / {total} sections…")

                    preview.markdown(
                        f"### Progress: {len(sections)}/{total} sections\n\n"
                        + "\n\n---\n\n".join(sections)
                    )
                    # 3. FIX: Only finalize when ALL sections have been generated
                    if len(sections) == total:
                        progress_bar.progress(100, text="Finalising…")
                        
                        final_doc = compile_handbook(topic, sections, plan_lines)
                        pdf_path = save_handbook_as_pdf(final_doc, topic)
                        
                        save_history_entry(
                            topic=topic,
                            final_doc=final_doc,
                            pdf_path=pdf_path,
                            sections=len(sections),
                        )
                        
                        progress_bar.empty()
                        st.success("✅ Handbook completed!")
                        preview.markdown(final_doc)
                        
                        with open(pdf_path, "rb") as fh:
                           st.download_button(
                           label="⬇️ Download PDF",
                           data=fh,
                           file_name=pdf_path.split("/")[-1],
                           mime="application/pdf",
                        )
        except ValueError as exc:
            st.error(f"Generation failed: {exc}")
        
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")