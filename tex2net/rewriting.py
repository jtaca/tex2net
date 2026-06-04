# character_interaction_graph/rewriting.py

from functools import lru_cache
import re
import warnings

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency in lightweight environments
    torch = None

@lru_cache(maxsize=4)
def _load_t5_components(model_name):
    """Load and cache the tokenizer/model pair used for abstractive summaries."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def _extract_interaction_summary(text):
    """Build a deterministic relation-focused summary without requiring Hugging Face downloads."""
    relations = []
    for sentence in [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]:
        people = re.findall(r"\b[A-Z][a-z]+\b", sentence)
        verbs = re.findall(r"\b[a-z]{3,}\b", sentence)
        if len(people) >= 2:
            action = verbs[0] if verbs else "interacts"
            relations.append(f"{people[0]} - {action} - {people[1]}")

    if relations:
        return "; ".join(relations)

    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    if sentences:
        return " ".join(sentences[:2])

    return text.strip() or "No summary available."


def summarize_t5(text, model_name="t5-small"):
    """
    Summarize text with T5 when available, otherwise fall back to a deterministic
    character-interaction summary.
    """
    if not text or not text.strip():
        return ""

    prompt = f"summarize character interactions: {text.strip()}"

    try:
        tokenizer, model = _load_t5_components(model_name)
        inputs = tokenizer(
            prompt,
            max_length=min(getattr(tokenizer, "model_max_length", 512), 512),
            truncation=True,
            return_tensors="pt",
        )

        if torch is not None and torch.cuda.is_available():
            model = model.to("cuda")
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        outputs = model.generate(
            **inputs,
            num_beams=4,
            max_length=128,
            min_length=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        return summary or _extract_interaction_summary(text)
    except Exception as exc:
        warnings.warn(
            f"Falling back to deterministic interaction summarization because the transformer model could not be loaded: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _extract_interaction_summary(text)

