"""
AI RFP Responder — PDF Parser module.

Extracts text from RFP PDFs and heuristically segments questions and
question–answer pairs for ingestion into the knowledge base.
"""

from __future__ import annotations

import io
import re
from typing import Dict, List

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize whitespace: collapse runs of whitespace to a single 
    space, strip."""
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf_text(file_bytes: bytes) -> List[Dict]:
    """Extract raw text from a PDF, page by page.

    Returns a list of dicts, one per page:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: List[Dict] = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text()
        if raw:
            pages.append({"page": i, "text": clean_text(raw)})
    return pages


def _find_question_starts(text: str) -> List[int]:
    """Return character offsets where questions typically start.

    Heuristic: numbered headings at line start, keyword markers,
    or a question mark followed by whitespace.
    """
    positions: List[int] = []
    # Numbered headings at line start: "1.", "1.1", "2.3", etc.
    for m in re.finditer(r"^\s*(\d+(?:\.\d+)*)\.", text, re.MULTILINE):
        positions.append(m.start())
    # Keyword markers
    for kw in ("Q:", "Question:", "Requirements:"):
        for m in re.finditer(re.escape(kw), text, re.IGNORECASE):
            positions.append(m.start())
    # Question mark boundaries: '?' followed by space
    for m in re.finditer(r"\?\s", text):
        positions.append(m.end())

    # De-duplicate and sort, ensuring minimum distance
    unique_positions: List[int] = []
    for pos in sorted(set(positions)):
        if not unique_positions or pos - unique_positions[-1] > 20:
            unique_positions.append(pos)

    return unique_positions


def segment_questions(page_texts: List[Dict]) -> List[Dict]:
    """Heuristically extract questions from a new (unanswered) RFP.

    For each page, find likely question start offsets and carve out the
    question text up to the next question start or a reasonable length limit.

    Returns a list of dicts:
        [{"page": int, "question": str, "section": str | None}, ...]
    """
    results: List[Dict] = []
    for page_info in page_texts:
        text = page_info.get("text", "")
        if not text:
            continue
        starts = _find_question_starts(text)
        if not starts:
            results.append(
                {
                    "page": page_info.get("page", 0),
                    "question": clean_text(text),
                    "section": None,
                }
            )
            continue
        for idx, start in enumerate(starts):
            if idx + 1 < len(starts):
                end = starts[idx + 1]
            else:
                end = min(start + 1000, len(text))
            question_text = clean_text(text[start:end])
            if question_text:
                results.append(
                    {
                        "page": page_info.get("page", 0),
                        "question": question_text,
                        "section": None,
                    }
                )
    return results


def segment_qa_pairs(
    page_texts: List[Dict], source_file: str
) -> List[Dict]:
    """Extract question–answer pairs from past answered RFPs (KB ingestion).

    Heuristic:
      - Find question-start markers the same way as segment_questions().
      - For each marker, the answer text is the segment from that marker
        up to the next question marker (or end of page text).
      - The question for each pair is the text between the previous marker
        and this one (or from start of text for the first pair).
      - Each pair carries metadata: source_file and page.

    Returns a list of dicts:
        [{"question": str, "answer": str, "page": int,
          "source": str, "chunk_id": str}, ...]
    """
    MAX_ANSWER_CHARS = 2000
    # Pre-extract question starts per page for efficiency
    page_starts: Dict[int, List[int]] = {}
    for page_info in page_texts:
        text = page_info.get("text", "")
        if text:
            page_starts[page_info.get("page", 0)] = _find_question_starts(
                text
            )
    results: List[Dict] = []
    for page_info in page_texts:
        page_num = page_info.get("page", 0)
        text = page_info.get("text", "")
        if not text or page_num not in page_starts:
            continue
        starts = page_starts[page_num]
        if not starts:
            results.append(
                {
                    "question": clean_text(text),
                    "answer": "",
                    "page": page_num,
                    "source": source_file,
                    "chunk_id": f"{source_file}_page{page_num}_qna",
                }
            )
            continue
        for idx, start in enumerate(starts):
            # Answer spans from this marker to the next marker (or end)
            if idx + 1 < len(starts):
                answer_end = starts[idx + 1]
            else:
                answer_end = len(text)
            raw_answer = text[start:answer_end]
            answer_text = clean_text(raw_answer)
            if len(answer_text) > MAX_ANSWER_CHARS:
                answer_text = (
                    clean_text(answer_text[:MAX_ANSWER_CHARS]) + " [...]"
                )
            # Question is the text between the previous marker and this one
            if idx == 0:
                question_start = 0
            else:
                question_start = starts[idx - 1]
            question_text = (
                clean_text(text[question_start:start])
                if question_start < start
                else ""
            )
            if question_text or answer_text:
                chunk_id = f"{source_file}_page{page_num}_q{idx}"
                results.append(
                    {
                        "question": question_text,
                        "answer": answer_text,
                        "page": page_num,
                        "source": source_file,
                        "chunk_id": chunk_id,
                    }
                )
    return results
