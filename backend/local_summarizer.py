"""Optional local-only summarization for retrieval, with deterministic fallback.

Set LOCAL_SUMMARIZER_MODEL to a locally cached Hugging Face seq2seq model to enable
model summarization. This module never downloads models or calls an external API.
"""
from __future__ import annotations

import os
import re
from typing import Any

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


def _load_local_model(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    model.eval()
    _MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def summarize_for_retrieval(text: str, max_chars: int = 2400) -> dict[str, str]:
    """Return a compact retrieval query; failures safely fall back to normalized text."""
    normalized = re.sub(r"\\s+", " ", str(text)).strip()[:max_chars]
    if not normalized:
        return {"summary": "", "mode": "deterministic-normalization"}
    model_name = os.getenv("LOCAL_SUMMARIZER_MODEL", "").strip()
    if model_name:
        try:
            tokenizer, model = _load_local_model(model_name)
            prompt = "Summarize this quantitative research idea, preserving data fields, operator intent, direction, and execution constraints: " + normalized
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            generated = model.generate(**encoded, max_new_tokens=128, do_sample=False)
            summary = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            if summary:
                return {"summary": summary[:max_chars], "mode": "local-pretrained"}
        except Exception:
            # Missing optional dependency/model files or inference failure must not
            # prevent research; no network/model download is attempted here.
            pass
    return {"summary": normalized, "mode": "deterministic-normalization"}
