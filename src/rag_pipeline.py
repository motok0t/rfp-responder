"""
AI RFP Responder — RAG Pipeline module.

Orchestrates the full RAG pipeline:
  Phase A (Knowledge Base): ingest past answered RFPs into ChromaDB
  Phase B (Process RFP): retrieve past answers and generate drafts for new RFPs
"""

from __future__ import annotations

from typing import Any, Dict, List

from src import config as cfg
from src import ollama_client, pdf_parser, vector_store


def ingest_rfp_pdf(file_bytes: bytes, source_file: str) -> Dict[str, Any]:
    """Orchestrate Phase A: ingest a past answered RFP PDF."""
    pages = pdf_parser.extract_pdf_text(file_bytes)

    if not pages:
        return {
            "chunks_indexed": 0,
            "errors": [f"No text extracted from {source_file}"],
        }

    q_pairs = pdf_parser.segment_qa_pairs(pages, source_file)

    if not q_pairs:
        return {
            "chunks_indexed": 0,
            "errors": [f"No Q&A pairs found in {source_file}"],
        }

    chunk_texts = []
    chunk_metadatas = []

    for pair in q_pairs:
        combined = "Q: " + pair["question"] + " A: " + pair["answer"]
        chunk_texts.append(combined)
        chunk_metadatas.append(
            {
                "source": pair["source"],
                "page": pair["page"],
                "question": pair["question"],
                "answer": pair["answer"],
            }
        )

    embeddings = ollama_client.embed_texts(
        chunk_texts, model=cfg.EMBEDDING_MODEL
    )

    chunks_for_upsert = []
    for i, pair in enumerate(q_pairs):
        chunks_for_upsert.append(
            {
                "id": pair["chunk_id"],
                "embedding": embeddings[i],
                "metadata": chunk_metadatas[i],
                "text": chunk_texts[i],
            }
        )

    vector_store.upsert_chunks(chunks_for_upsert, embeddings)

    return {
        "chunks_indexed": len(q_pairs),
        "source_file": source_file,
        "errors": [],
    }


def retrieve_context(
    question: str, top_k: int | None = None
) -> List[Dict[str, Any]]:
    """Retrieve the top-k most relevant past Q&A pairs."""
    k = top_k or cfg.TOP_K
    question_embedding = ollama_client.embed_text(
        question, model=cfg.EMBEDDING_MODEL
    )
    result = vector_store.query_similar(question_embedding, top_k=k)

    matches = []
    if result.get("ids") and result["ids"][0]:
        ids = result["ids"][0]
        distances = result.get("distances", [[1.0] * len(ids)])[0]
        metadatas = result.get("metadatas", [[] for _ in ids])[0]

        for i, doc_id in enumerate(ids):
            match = {
                "question": metadatas[i].get("question", "")
                if metadatas[i]
                else "",
                "answer": metadatas[i].get("answer", "")
                if metadatas[i]
                else "",
                "source": metadatas[i].get("source", "")
                if metadatas[i]
                else "",
                "page": metadatas[i].get("page", 0) if metadatas[i] else 0,
                "similarity_score": 1 - distances[i] if distances[i] else 0.0,
            }
            matches.append(match)

    return matches


def generate_draft(
    question: str, retrieved_context: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate a draft answer for a single question using RAG."""
    prompt = ollama_client.build_rag_prompt(question, retrieved_context)
    draft = ollama_client.generate_answer(
        prompt=prompt, model=cfg.GENERATION_MODEL, temperature=0.0
    )

    sources = []
    for item in retrieved_context:
        src = item.get("source", "")
        pg = item.get("page", 0)
        if src and pg is not None:
            source_str = f"{src} (page {pg})"
            if source_str not in sources:
                sources.append(source_str)

    return {"question": question, "draft": draft, "sources": sources}


def process_rfp_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Orchestrate Phase B: process a new RFP PDF and generate drafts."""
    pages = pdf_parser.extract_pdf_text(file_bytes)
    if not pages:
        return [
            {
                "question": "",
                "draft": "No text could be extracted from the RFP PDF.",
                "sources": [],
            }
        ]

    questions_data = pdf_parser.segment_questions(pages)

    if not questions_data:
        return [
            {
                "question": "",
                "draft": "No questions found in the RFP PDF.",
                "sources": [],
            }
        ]

    drafts = []
    for q_info in questions_data:
        question_text = q_info.get("question", "")

        if not question_text.strip():
            drafts.append({"question": "", "draft": "", "sources": []})
            continue

        context = retrieve_context(question_text)
        draft_result = generate_draft(question_text, context)
        drafts.append(draft_result)

    return drafts
