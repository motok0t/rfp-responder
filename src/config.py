"""
AI RFP Responder — Configuration

Central constants for the app. Adjust these if you pull different Ollama models
or want to tune the RAG pipeline.
"""

# --- Ollama ---
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GENERATION_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "rfp_knowledge_base"
TOP_K = 5  # number of past Q&A pairs to retrieve per question

# --- Chunking / Ingest ---
MAX_CHUNK_CHARS = 2000  # hard cap for the text sent to the LLM per chunk
MIN_CHUNK_CHARS = 50    # discard chunks shorter than this after segmentation

# --- Prompt template (RAG) ---
RAG_PROMPT_TEMPLATE = (
    "You are an expert B2B proposal writer. Using ONLY the PAST EXAMPLES below as reference "
    "context, draft a concise, professional answer to the NEW QUESTION. Match the tone, detail "
    "level, and substance of the examples. If the examples do not contain enough information to "
    "answer fully, state what is missing — do not invent facts.\n\n"
    "PAST EXAMPLES:\n"
    "{retrieved_context}\n\n"
    "NEW QUESTION:\n"
    "{question}\n\n"
    "DRAFT ANSWER:\n"
)
