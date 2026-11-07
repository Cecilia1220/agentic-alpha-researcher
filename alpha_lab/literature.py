"""Bounded lexical retrieval over local, attributed literature passages."""
import re
from datetime import date


def terms(text):
    return set(re.findall(r"[a-z][a-z_0-9]+", text.lower()))


def validate_corpus(passages):
    if not isinstance(passages, list):
        raise ValueError("literature corpus must be a list of passages")
    ids = set()
    for passage in passages:
        required = {"id", "title", "source", "available_on", "keywords", "note"}
        if not isinstance(passage, dict) or passage.keys() != required:
            raise ValueError("literature passages need id, title, source, available_on, keywords and note")
        for key in required - {"keywords"}:
            value = passage[key]
            if not isinstance(value, str) or not value.strip() or len(value) > 900:
                raise ValueError(f"invalid literature field {key}")
        date.fromisoformat(passage["available_on"])
        keywords = passage["keywords"]
        if (not isinstance(keywords, list) or len(keywords) > 20
                or any(not isinstance(word, str) or len(word) > 60 for word in keywords)):
            raise ValueError("keywords must be a list of short strings")
        if passage["id"] in ids:
            raise ValueError("duplicate literature passage ID")
        ids.add(passage["id"])
    return passages


def retrieve(passages, query, as_of, *, limit=2):
    """Prioritize field-specific keyword matches, then passage text. Return only hits."""
    query_terms = terms(query) - {
        "the", "for", "and", "with", "into", "may", "have", "are", "using", "from", "that",
        "test", "research", "hypothesis", "daily", "cross", "sectional", "relative", "return", "returns", "stock", "equity",
    }
    ranked = []
    for passage in passages:
        if date.fromisoformat(passage["available_on"]) >= as_of:
            continue
        keyword_terms = terms(" ".join(passage["keywords"]))
        text_terms = terms(passage["title"] + " " + passage["note"])
        score = 3 * len(query_terms & keyword_terms) + len(query_terms & text_terms)
        if score:
            ranked.append((score, passage))
    ranked.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [passage for _, passage in ranked[:limit]]
