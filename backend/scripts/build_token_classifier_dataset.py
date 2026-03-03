#!/usr/bin/env python3
"""Build keep/drop token labels from optimization_history."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - environment dependent
    AutoTokenizer = None

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Record:
    raw_prompt: str
    optimized_prompt: str
    optimization_id: str


def _resolve_db_path(db_path: str | None) -> Path:
    if db_path:
        return Path(db_path)
    try:
        from database import DB_PATH  # noqa: WPS433
    except Exception:
        return Path(__file__).resolve().parents[1] / "app.db"
    return Path(DB_PATH)


def _fetch_records(
    connection: sqlite3.Connection, limit: int | None
) -> Iterable[Record]:
    query = """
        SELECT id, raw_prompt, optimized_prompt
        FROM optimization_history
        ORDER BY created_at DESC
    """
    if limit:
        query += " LIMIT ?"
        params: Sequence[object] = (limit,)
    else:
        params = ()
    cursor = connection.execute(query, params)
    for row in cursor:
        raw_prompt = row[1] if row[1] is not None else ""
        optimized_prompt = row[2] if row[2] is not None else ""
        if not raw_prompt.strip() or not optimized_prompt.strip():
            continue
        yield Record(
            raw_prompt=raw_prompt,
            optimized_prompt=optimized_prompt,
            optimization_id=row[0],
        )


def _matching_spans(source: str, target: str) -> List[Tuple[int, int]]:
    matcher = SequenceMatcher(None, source, target)
    spans: List[Tuple[int, int]] = []
    for match in matcher.get_matching_blocks():
        if match.size <= 0:
            continue
        spans.append((match.a, match.a + match.size))
    return spans


def _overlaps(span: Tuple[int, int], spans: Sequence[Tuple[int, int]]) -> bool:
    start, end = span
    for span_start, span_end in spans:
        if start < span_end and end > span_start:
            return True
    return False


def _label_tokens(
    tokenizer,
    text: str,
    keep_spans: Sequence[Tuple[int, int]],
    max_length: int,
) -> dict:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise ValueError("Tokenizer did not return offset mappings.")
    labels: List[int] = []
    for start, end in encoded["offset_mapping"]:
        if start == 0 and end == 0:
            labels.append(-100)
            continue
        label = 1 if _overlaps((start, end), keep_spans) else 0
        labels.append(label)
    encoded.pop("offset_mapping", None)
    encoded["labels"] = labels
    return encoded


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build keep/drop token labels from optimization_history."
    )
    parser.add_argument("--db-path", help="Path to sqlite db (defaults to DB_PATH).")
    parser.add_argument(
        "--model-name",
        default="microsoft/MiniLM-L12-H384-uncased",
        help="Tokenizer model to align labels with.",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/data/token_classifier",
        help="Directory to store train/eval JSONL files.",
    )
    parser.add_argument("--limit", type=int, help="Max rows to load.")
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows to reserve for evaluation.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.85,
        help="Skip examples with lower raw/optimized similarity.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=384,
        help="Tokenizer max length for generated labels.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/eval split."
    )
    args = parser.parse_args()

    if AutoTokenizer is None:
        raise SystemExit(
            "transformers is required. Install backend/requirements.txt first."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit(
            "Tokenizer must be a fast tokenizer to provide offset mappings."
        )
    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    LOGGER.info("Using database at %s", db_path)
    with sqlite3.connect(str(db_path)) as connection:
        connection.row_factory = sqlite3.Row
        records = list(_fetch_records(connection, args.limit))
        if not records:
            raise SystemExit("No optimization_history rows found.")

    random.seed(args.seed)
    random.shuffle(records)
    eval_size = int(len(records) * args.eval_ratio)
    eval_records = records[:eval_size]
    train_records = records[eval_size:]

    train_rows: List[dict] = []
    eval_rows: List[dict] = []
    skipped = 0
    for bucket, rows in ((train_rows, train_records), (eval_rows, eval_records)):
        for record in rows:
            similarity = SequenceMatcher(
                None, record.raw_prompt, record.optimized_prompt
            ).ratio()
            if similarity < args.min_similarity:
                skipped += 1
                continue
            keep_spans = _matching_spans(record.raw_prompt, record.optimized_prompt)
            encoded = _label_tokens(
                tokenizer, record.raw_prompt, keep_spans, args.max_length
            )
            encoded["optimization_id"] = record.optimization_id
            encoded["text"] = record.raw_prompt
            bucket.append(encoded)

    output_dir = Path(args.output_dir)
    _write_jsonl(output_dir / "train.jsonl", train_rows)
    _write_jsonl(output_dir / "eval.jsonl", eval_rows)

    LOGGER.info(
        "Wrote %s train rows and %s eval rows (%s skipped).",
        len(train_rows),
        len(eval_rows),
        skipped,
    )


if __name__ == "__main__":
    main()
