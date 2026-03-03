from __future__ import annotations

import re
from typing import Any, Dict, List, Match, Optional, Set

from . import config

ROLE_PREFIX_PATTERN = re.compile(
    r"(user|assistant|system|developer|tool):",
    flags=re.IGNORECASE,
)

_DIRECTIVE_KEYWORD_ALTERNATION = "|".join(
    re.escape(keyword) for keyword in config.DIRECTIVE_KEYWORDS
)

DIRECTIVE_KEYWORD_PATTERN = (
    re.compile(r"\b(?:" + _DIRECTIVE_KEYWORD_ALTERNATION + r")\b", flags=re.IGNORECASE)
    if _DIRECTIVE_KEYWORD_ALTERNATION
    else None
)

DIRECTIVE_SNIPPET_PATTERN = (
    re.compile(
        r"[^.!?\n]*(?:" + _DIRECTIVE_KEYWORD_ALTERNATION + r")[^.!?\n]*",
        flags=re.IGNORECASE,
    )
    if _DIRECTIVE_KEYWORD_ALTERNATION
    else None
)

FALLBACK_SNIPPET_PATTERN = re.compile(r"[^.!?\n]+")


def summarize_history(opt, text: str) -> str:
    """Summarize long conversation history by scoring messages and preserving
    the highest-value turns while compressing the remainder into directive notes.
    """
    boundary_chars = {".", "!", "?", ";", ":", "]", ")", "}", '"', "'", ">"}

    matches: List[Match[str]] = []
    for match in ROLE_PREFIX_PATTERN.finditer(text):
        start = match.start()
        line_start = text.rfind("\n", 0, start) + 1
        preceding_segment = text[line_start:start]

        if preceding_segment.strip():
            trimmed = preceding_segment.rstrip()
            prev_char = trimmed[-1] if trimmed else ""
            if prev_char not in boundary_chars:
                continue

        matches.append(match)

    if len(matches) <= 5:
        return text

    scored_messages = []
    total_messages = len(matches)
    total_words = 0

    for index, match in enumerate(matches):
        role = match.group(1)
        content_start = match.end()
        content_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(text)
        )
        content = text[content_start:content_end]
        total_words += len(content.split())
        score = score_history_turn(opt, role, content, index, total_messages)
        scored_messages.append(
            {
                "role": role,
                "content": content,
                "index": index,
                "start": match.start(),
                "end": content_end,
                "score": score,
            }
        )

    avg_message_length = int(total_words / total_messages) if total_messages else 0
    has_system_prompt = any(msg["role"].lower() == "system" for msg in scored_messages)
    keep_ratio = calculate_keep_ratio_adaptive(
        total_messages=total_messages,
        has_system_prompt=has_system_prompt,
        avg_message_length=avg_message_length,
    )
    summarize_modifier = getattr(opt, "summarize_keep_ratio_modifier", 1.0)
    if summarize_modifier != 1.0:
        keep_ratio = min(0.95, max(0.1, keep_ratio * summarize_modifier))
    keep_count = max(4, int(total_messages * keep_ratio))
    if keep_count >= total_messages:
        return text

    sorted_by_score = sorted(
        scored_messages, key=lambda item: item["score"], reverse=True
    )
    keep_indices = {item["index"] for item in sorted_by_score[:keep_count]}

    for message in scored_messages:
        if message["role"].lower() == "system":
            keep_indices.add(message["index"])

    # Always keep the latest exchange so the optimizer has the current context.
    last_message_index = scored_messages[-1]["index"]
    keep_indices.add(last_message_index)

    last_user_message = next(
        (msg for msg in reversed(scored_messages) if msg["role"].lower() == "user"),
        None,
    )
    if last_user_message is not None:
        keep_indices.add(last_user_message["index"])

        next_assistant_after_user = next(
            (
                msg
                for msg in scored_messages
                if msg["index"] > last_user_message["index"]
                and msg["role"].lower() == "assistant"
            ),
            None,
        )
        if next_assistant_after_user is not None:
            keep_indices.add(next_assistant_after_user["index"])

    # Preserve user/assistant exchanges together so prompts remain paired with replies.
    pairing_changed = True
    while pairing_changed:
        pairing_changed = False
        for current_position, message in enumerate(scored_messages):
            role = message["role"].lower()
            if role not in {"user", "assistant"}:
                continue

            if message["index"] not in keep_indices:
                continue

            if role == "user":
                partner = next(
                    (
                        candidate
                        for candidate in scored_messages[current_position + 1 :]
                        if candidate["role"].lower() == "assistant"
                    ),
                    None,
                )
            else:  # assistant
                partner = next(
                    (
                        candidate
                        for candidate in reversed(scored_messages[:current_position])
                        if candidate["role"].lower() == "user"
                    ),
                    None,
                )

            if partner is None:
                continue

            if partner["index"] not in keep_indices:
                keep_indices.add(partner["index"])
                pairing_changed = True

    kept_messages = [msg for msg in scored_messages if msg["index"] in keep_indices]
    kept_messages.sort(key=lambda item: item["index"])

    discarded_messages = [
        msg for msg in scored_messages if msg["index"] not in keep_indices
    ]
    notes = build_history_notes(opt, discarded_messages)

    conversation_lines: List[str] = []
    for message in kept_messages:
        content = message["content"].strip()
        line = f"{message['role']}: {content}" if content else f"{message['role']}:"
        conversation_lines.append(line)

    if notes:
        conversation_lines.append("Summary Notes:")
        conversation_lines.extend(notes)

    summarized_conv = "\n".join(conversation_lines)
    if summarized_conv and not summarized_conv.endswith("\n"):
        summarized_conv += "\n"

    conv_start = matches[0].start()
    conv_end = scored_messages[-1]["end"]

    return text[:conv_start] + summarized_conv + text[conv_end:]


def score_history_turn(opt, role: str, content: str, index: int, total: int) -> float:
    role_weight = config.ROLE_WEIGHTS.get(role.lower(), 1.0)

    keyword_matches = (
        DIRECTIVE_KEYWORD_PATTERN.findall(content)
        if DIRECTIVE_KEYWORD_PATTERN is not None
        else []
    )
    keyword_bonus = len(keyword_matches) * 1.5

    recency_bonus = 1.0
    if total > 0:
        recency_bonus += (index + 1) / total * 1.5

    return role_weight + keyword_bonus + recency_bonus


def calculate_keep_ratio_adaptive(
    *,
    total_messages: int,
    has_system_prompt: bool,
    avg_message_length: int,
) -> float:
    base_ratio = 0.4

    if total_messages > 50:
        base_ratio = 0.25
    elif total_messages > 20:
        base_ratio = 0.3

    if avg_message_length < 50:
        base_ratio *= 1.2

    if has_system_prompt:
        base_ratio = max(base_ratio, 0.3)

    min_keep = 8 / total_messages if total_messages > 8 else 0.5
    return max(base_ratio, min_keep)


def build_history_notes(opt, messages: List[Dict]) -> List[str]:
    if not messages:
        return []

    notes: List[str] = []
    seen: Set[str] = set()

    for message in messages:
        content = " ".join(message["content"].split())
        if not content:
            continue

        snippet_match = (
            DIRECTIVE_SNIPPET_PATTERN.search(content)
            if DIRECTIVE_SNIPPET_PATTERN is not None
            else None
        )
        if snippet_match:
            snippet = snippet_match.group(0)
        else:
            fallback_match = FALLBACK_SNIPPET_PATTERN.search(content)
            snippet = fallback_match.group(0) if fallback_match else ""

        snippet = snippet.strip()
        if not snippet:
            continue

        if len(snippet) > 160:
            snippet = snippet[:157].rstrip() + "..."

        note = f"- {message['role'].title()}: {snippet}"
        if note.lower() in seen:
            continue

        notes.append(note)
        seen.add(note.lower())

    return notes


def parse_chat_segments(text: str) -> List[Dict[str, Optional[str]]]:
    if not text.strip():
        return []

    pattern = re.compile(r"^\s*([A-Za-z0-9_\- ]+):", flags=re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [{"role": None, "content": text.strip()}]

    segments: List[Dict[str, Optional[str]]] = []

    for index, match in enumerate(matches):
        role = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment_body = text[start:end].strip()
        segments.append({"role": role, "content": segment_body})

    return segments


def restore_structured_chat(metadata: Dict[str, Any], optimized_text: str) -> str:
    """Rebuild the chat transcript ensuring protected roles remain verbatim."""

    order = metadata.get("order") or []
    if not order:
        return optimized_text

    if metadata.get("skip_roles"):
        return optimized_text

    segments = parse_chat_segments(optimized_text)
    segment_index = 0
    restored_lines: List[str] = []

    for entry in order:
        role = entry["role"]
        if entry.get("protected"):
            content = entry.get("content", "")
            restored_lines.append(f"{role}: {content}".rstrip())
            if (
                segment_index < len(segments)
                and (segments[segment_index]["role"] or "").lower() == role.lower()
            ):
                segment_index += 1
            continue

        content = ""
        if segment_index < len(segments):
            segment = segments[segment_index]
            segment_index += 1
            content = (segment.get("content") or "").strip()

        if content:
            restored_lines.append(f"{role}: {content}".rstrip())

    residual_segments = segments[segment_index:]
    for segment in residual_segments:
        role = segment.get("role")
        content = (segment.get("content") or "").strip()
        if role:
            restored_lines.append(f"{role}: {content}".rstrip())
        elif content:
            restored_lines.append(content)

    final_text = "\n".join(line for line in restored_lines if line).strip()
    return final_text
