from __future__ import annotations

import re
from typing import Optional, Set


def build_coref_alias(reference: str, reserved_aliases: Set[str]) -> Optional[str]:
    text = reference.strip()
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return None

    def reserve_alias(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        base_alias = candidate
        alias = candidate
        suffix = 2
        while alias.lower() in reserved_aliases:
            alias = f"{base_alias}{suffix}"
            suffix += 1
            if suffix > 9:
                return None
        reserved_aliases.add(alias.lower())
        return alias

    # Strategy 1: acronym for multi-word entities
    if len(words) >= 2:
        acronym = "".join(word[0].upper() for word in words if word[0].isalpha())
        if len(acronym) >= 2 and len(acronym) < len(text):
            reserved = reserve_alias(acronym)
            if reserved:
                return reserved

    # Strategy 2: use last name for TitleCase two-word names
    if len(words) == 2 and all(word[0].isupper() for word in words):
        last_name = words[-1]
        if len(last_name) < len(text):
            reserved = reserve_alias(last_name)
            if reserved:
                return reserved

    # Strategy 3: titled entities
    title_map = {
        "doctor": "Dr",
        "professor": "Prof",
        "mister": "Mr",
        "missus": "Mrs",
        "miss": "Ms",
    }
    lower_words = [word.lower() for word in words]
    for title, short in title_map.items():
        if title in lower_words:
            idx = lower_words.index(title)
            if idx < len(words) - 1:
                alias = f"{short} {words[idx + 1]}"
                if len(alias) < len(text):
                    reserved = reserve_alias(alias)
                    if reserved:
                        return reserved

    # Strategy 4: truncated fallback for long references
    if len(text) > 15:
        truncated = text[:12].rstrip() + "..."
        if len(truncated) < len(text):
            return reserve_alias(truncated)

    return None


def select_coref_pronoun(reference: str) -> str:
    if not reference:
        return ""

    normalized = reference.lower().strip()
    plural_keywords = {
        "team",
        "group",
        "people",
        "users",
        "developers",
        "engineers",
        "stakeholders",
        "participants",
        "members",
    }

    if any(word in normalized for word in plural_keywords):
        return "they"

    singular_keywords = {
        "system",
        "application",
        "service",
        "assistant",
        "model",
        "tool",
        "api",
    }

    if any(word in normalized for word in singular_keywords):
        return "it"

    # Proper names and organizations: prefer neutral "they" over "it"
    # Heuristic: all-caps or TitleCase tokens (e.g., "ALICE", "ACME Inc.", "Alice Smith")
    ref = reference.strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", ref)
    if ref.isupper() or (tokens and all(t[0].isupper() for t in tokens)):
        return "they"

    # Plural morphology fallback
    if normalized.endswith("s"):
        return "they"

    return "it"


__all__ = ["build_coref_alias", "select_coref_pronoun"]
