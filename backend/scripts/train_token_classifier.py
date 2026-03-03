#!/usr/bin/env python3
"""Fine-tune a compact token classifier for keep/drop labeling."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
except ImportError:  # pragma: no cover - environment dependent
    load_dataset = None
    AutoModelForTokenClassification = None
    AutoTokenizer = None
    DataCollatorForTokenClassification = None
    Trainer = None
    TrainingArguments = None

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a token classifier using keep/drop labels."
    )
    parser.add_argument(
        "--train-data",
        default="backend/data/token_classifier/train.jsonl",
        help="Path to JSONL training data.",
    )
    parser.add_argument(
        "--eval-data",
        default="backend/data/token_classifier/eval.jsonl",
        help="Path to JSONL evaluation data.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/MiniLM-L12-H384-uncased",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/models/token_classifier",
        help="Directory to write fine-tuned model artifacts.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if load_dataset is None:
        raise SystemExit(
            "datasets/transformers are required. Install backend/requirements.txt."
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    label_list = ["drop", "keep"]
    label2id: Dict[str, int] = {"drop": 0, "keep": 1}
    id2label = {value: key for key, value in label2id.items()}

    train_path = Path(args.train_data)
    eval_path = Path(args.eval_data)
    if not train_path.exists():
        raise SystemExit(f"Training data not found: {train_path}")
    if not eval_path.exists():
        raise SystemExit(f"Evaluation data not found: {eval_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
    )

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "eval": str(eval_path)},
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=args.logging_steps,
        save_steps=args.logging_steps * 5,
        eval_steps=args.logging_steps * 5,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    LOGGER.info("Starting fine-tuning on %s", args.model_name)
    trainer.train()
    LOGGER.info("Saving model to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
