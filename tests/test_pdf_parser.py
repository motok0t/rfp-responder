"""Unit tests for src/pdf_parser.py."""

from src.pdf_parser import clean_text, segment_questions, segment_qa_pairs


def test_clean_text():
    """Test whitespace normalization."""
    assert clean_text("  Hello   world! \n") == "Hello world!"
    assert clean_text("\n\n   \t  ") == ""


def test_segment_questions():
    """Test extraction of questions from raw text."""
    sample_text = [
        {
            "page": 1,
            "text": "1. What is your focus?\n2. What is your security?",
        }
    ]

    result = segment_questions(sample_text)

    # We expect 2 questions to be extracted based on line breaks
    assert len(result) == 2
    assert "What is your focus" in result[0]["question"]
    assert "What is your security" in result[1]["question"]


def test_segment_qa_pairs():
    """Test extraction of question-answer pairs."""
    sample_text = [
        {
            "page": 1,
            "text": (
                "1. What is your focus? "
                "We focus on healthcare. "
                "2. What is your security? "
                "We use AES-256."
            ),
        }
    ]

    result = segment_qa_pairs(sample_text, "test.pdf")

    assert len(result) >= 1
    assert "test.pdf" in result[0]["source"]
    assert result[0]["chunk_id"].startswith("test.pdf")
