"""
AI RFP Responder — Ollama client wrapper.

Provides:
  - ollama_available()          -> bool
  - ensure_model(model_name)    -> bool
  - embed_text(text, model)     -> list[float]
  - embed_texts(texts, model)   -> list[list[float]]
  - generate_answer(prompt)     -> str
  - build_rag_prompt(...)       -> str
"""

from __future__ import annotations

from typing import List, Optional

import requests

from src import config as cfg


def _ollama_get_tags(base_url: str = cfg.OLLAMA_BASE_URL) -> dict:
    """GET /api/tags — return the parsed JSON response."""
    url = f"{base_url.rstrip('/')}/api/tags"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def ollama_available(base_url: str = cfg.OLLAMA_BASE_URL) -> bool:
    """Return True if the Ollama service is reachable and has models."""
    try:
        tags = _ollama_get_tags(base_url)
        if not tags or "models" not in tags:
            return False
        return isinstance(tags["models"], list)
    except Exception:
        return False


def ensure_model(
    model_name: str,
    base_url: str = cfg.OLLAMA_BASE_URL,
    timeout: float = 300.0,
) -> bool:
    """Pull *model_name* from Ollama if it isn't already available."""
    try:
        tags = _ollama_get_tags(base_url)
        models = tags.get("models", [])
        model_names = [m.get("name") for m in models]
        if model_name in model_names:
            return True
    except Exception:
        pass

    try:
        _ollama_post(
            "/api/pull",
            {"model": model_name, "stream": False},
            base_url=base_url,
            timeout=timeout,
        )
        tags = _ollama_get_tags(base_url)
        model_names = [m.get("name") for m in tags.get("models", [])]
        return model_name in model_names
    except Exception as e:
        print(f"[ollama_client] Failed to ensure model '{model_name}': {e}")
        return False


def _ollama_post(
    endpoint: str,
    payload: dict,
    base_url: str = cfg.OLLAMA_BASE_URL,
    timeout: float = 120.0,
) -> dict:
    """Thin wrapper around Ollama's HTTP API (POST)."""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def embed_text(
    text: str,
    model: str = cfg.EMBEDDING_MODEL,
    base_url: str = cfg.OLLAMA_BASE_URL,
) -> List[float]:
    """Embed a single text string using Ollama's embedding endpoint."""
    payload = {"model": model, "input": [text]}
    data = _ollama_post(
        "/api/embed", payload, base_url=base_url, timeout=60.0
    )
    emb = data.get("embeddings")
    if not emb or not emb[0]:
        raise ValueError(f"Unexpected embed response: {data}")
    return emb[0]


def embed_texts(
    texts: List[str],
    model: str = cfg.EMBEDDING_MODEL,
    base_url: str = cfg.OLLAMA_BASE_URL,
    batch_size: int = 32,
) -> List[List[float]]:
    """Embed a list of texts, processing in batches."""
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        for txt in batch:
            vectors.append(embed_text(txt, model=model, base_url=base_url))
    return vectors


def generate_answer(
    prompt: str,
    model: str = cfg.GENERATION_MODEL,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    base_url: str = cfg.OLLAMA_BASE_URL,
    system_prompt: Optional[str] = None,
) -> str:
    """Send a prompt to Ollama's chat API and return the raw answer text."""
    messages: List[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    options: dict = {
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    data = _ollama_post(
        "/api/chat", payload, base_url=base_url, timeout=120.0
    )

    msg = data.get("message", {})
    content = msg.get("content")
    if content is None:
        raise ValueError(f"Unexpected chat response, no content: {data}")
    return content.strip()


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[dict],
    prompt_template: str = "",
) -> str:
    """Assemble the few-shot prompt from retrieved past Q&A."""
    if not prompt_template:
        prompt_template = (
            "You are an expert B2B proposal writer. Using ONLY the PAST "
            "EXAMPLES below, draft a concise, professional answer to the "
            "NEW QUESTION.\n"
            "RULES:\n"
            "- Return ONLY the drafted text.\n"
            "- DO NOT write introductory phrases like 'Based on...'.\n"
            "- DO NOT use prefixes like 'A:' or 'DRAFT ANSWER:'.\n"
            "- If the examples do not contain enough information, state that."
            "\n\n"
            "PAST EXAMPLES:\n"
            "{retrieved_context}\n\n"
            "NEW QUESTION:\n"
            "{question}\n\n"
            "ANSWER:\n"
        )

    context_parts: List[str] = []
    for i, c in enumerate(retrieved_chunks, start=1):
        ctx = (
            f"Example {i} (source: {c.get('source', '?')}, "
            f"page {c.get('page', '?')}):\n"
        )
        ctx += f"Q: {c.get('question', '')}\n"
        ctx += f"A: {c.get('answer', '')}\n"
        context_parts.append(ctx)

    retrieved_context = "\n".join(context_parts)
    return prompt_template.format(
        retrieved_context=(
            retrieved_context if context_parts else "(no past examples found)"
        ),
        question=question,
    )
