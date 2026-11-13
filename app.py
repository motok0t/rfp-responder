"""
AI RFP Responder — Streamlit Application.

B2B MVP: AI-powered RFP (Request for Proposal) responder.
Two main tabs: Knowledge Base and Process RFP.
"""

from typing import List

import streamlit as st

from src.ollama_client import ollama_available
from src.rag_pipeline import ingest_rfp_pdf, process_rfp_pdf


def format_sources(sources: List[str]) -> str:
    """Return a markdown-formatted string of sources."""
    if not sources:
        return "_No sources_"
    return "\n".join(f"- {s}" for s in sources)


st.sidebar.title("AI RFP Responder")
st.sidebar.markdown(
    "An AI-powered tool that helps B2B companies generate draft responses "
    "for Requests for Proposals (RFPs) using a local RAG pipeline."
)
st.sidebar.markdown(
    "**How to use:**\n"
    "1. Go to 'Knowledge Base' and upload past answered RFPs.\n"
    "2. Go to 'Process RFP' and upload a new RFP to generate drafts."
)
st.sidebar.divider()

st.sidebar.title("Status")
if ollama_available():
    st.sidebar.success("Ollama connected")
    st.sidebar.caption(
        "Models: llama3.2:3b (gen), nomic-embed-text (embed)"
    )
else:
    st.sidebar.error("Ollama not reachable")
    st.sidebar.info(
        "Start Ollama (https://github.com/ollama/ollama) and run:\n"
        "  ollama pull llama3.2:3b\n"
        "  ollama pull nomic-embed-text"
    )


def kb_tab() -> None:
    st.header("Knowledge Base")
    st.file_uploader(
        "Upload past answered RFPs (PDFs) to populate the knowledge base:",
        type=["pdf"],
        accept_multiple_files=True,
        key="kb_uploader",
    )

    if st.button("Index Documents", type="primary", key="index_btn"):
        files = st.session_state.get("kb_uploader", [])
        if not files:
            st.warning("No files uploaded.")
            return

        progress = st.progress(0, text="Starting indexing...")
        total_indexed = 0

        for i, uploaded_file in enumerate(files):
            file_bytes = uploaded_file.read()
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                result = ingest_rfp_pdf(file_bytes, uploaded_file.name)
            total_indexed += result.get("chunks_indexed", 0)

            progress.progress(
                (i + 1) / len(files),
                text=f"Indexed {i + 1}/{len(files)} files",
            )

        progress.empty()
        st.success(f"Indexing complete! {total_indexed} chunks added.")


def process_rfp_tab() -> None:
    st.header("Process RFP")
    st.file_uploader(
        "Upload a new RFP PDF:",
        type=["pdf"],
        accept_multiple_files=False,
        key="rfp_uploader",
    )

    if st.button("Generate Drafts", type="primary", key="generate_btn"):
        uploaded = st.session_state.get("rfp_uploader")
        if not uploaded:
            st.warning("No RFP file uploaded.")
            return

        with st.spinner("Processing RFP..."):
            file_bytes = uploaded.read()
            result = process_rfp_pdf(file_bytes)
        st.session_state["rfp_drafts"] = result
        st.rerun()

    drafts = st.session_state.get("rfp_drafts", [])

    if not drafts:
        st.info("No drafts yet. Click 'Generate Drafts' above.")
        return

    st.success(f"Generated {len(drafts)} draft answer(s).")

    for i, draft in enumerate(drafts):
        st.subheader(f"Question {i + 1}")
        st.write(draft["question"])

        st.text_area(
            f"Draft answer {i + 1}",
            value=draft["draft"],
            key=f"draft_editor_{i}",
            height=150,
        )

        st.caption("Sources:")
        st.write(format_sources(draft["sources"]))

    if st.button("Download All Drafts as TXT", key="download_btn"):
        lines: List[str] = []
        for i, draft in enumerate(drafts, start=1):
            lines.append(f"=== Question {i} ===")
            lines.append(draft["question"])
            lines.append("")
            lines.append(f"=== Draft Answer {i} ===")
            lines.append(draft["draft"])
            lines.append("")
            lines.append(f"=== Sources {i} ===")
            lines.append(format_sources(draft["sources"]))
            lines.append("")

        content = "\n".join(lines)
        st.download_button(
            label="Download TXT",
            data=content,
            file_name="rfp_drafts.txt",
            mime="text/plain",
            key="download_txt",
        )


def main() -> None:
    tab1, tab2 = st.tabs(["Knowledge Base", "Process RFP"])
    with tab1:
        kb_tab()
    with tab2:
        process_rfp_tab()


if __name__ == "__main__":
    main()
