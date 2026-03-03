"""Unit tests for entropy scoring helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
from services.optimizer import entropy


@pytest.mark.skipif(
    entropy.torch is None or entropy.F is None, reason="torch not available"
)
def test_entropy_scoring_batches_multi_token_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure score_tokens batches candidates for multi-token prompts."""

    torch = entropy.torch
    assert torch is not None  # for mypy

    class DummyTokenizer:
        def __init__(self) -> None:
            self.all_special_ids = {0, 1}
            self.vocab = {
                "alpha": 2,
                "beta": 3,
                "gamma": 4,
                "delta": 5,
            }
            self.model_max_length = 16

        def __call__(
            self,
            text: str,
            return_offsets_mapping: bool = True,
            return_tensors: str = "pt",
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_overflowing_tokens: bool = False,
        ):
            tokens = text.split()
            offsets = []
            cursor = 0
            for token in tokens:
                start = text.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end + 1

            ids = torch.tensor(
                [self.vocab[token] for token in tokens], dtype=torch.long
            )
            attention = torch.ones_like(ids)
            mapping = torch.tensor(offsets, dtype=torch.long)
            return {
                "input_ids": ids.unsqueeze(0),
                "attention_mask": attention.unsqueeze(0),
                "offset_mapping": mapping.unsqueeze(0),
            }

    class DummyModel(torch.nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.forward_inputs = []

        def forward(self, input_ids, attention_mask=None):  # type: ignore[override]
            self.forward_inputs.append(input_ids.clone())
            batch, seq = input_ids.shape
            base = torch.linspace(
                0.0, 1.0, steps=self.vocab_size, device=input_ids.device
            )
            logits = base.repeat(batch * seq).view(batch, seq, self.vocab_size)
            return SimpleNamespace(logits=logits)

    dummy_tokenizer = DummyTokenizer()
    dummy_model = DummyModel(vocab_size=32)

    monkeypatch.setattr(
        entropy,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: dummy_tokenizer),
    )
    monkeypatch.setattr(
        entropy,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: dummy_model),
    )
    monkeypatch.setattr(
        entropy,
        "resolve_cached_model_path",
        lambda *_args, **_kwargs: "fake-model-path",
    )
    monkeypatch.setattr(entropy, "_SCORER", None)

    model = entropy._EntropyModel()
    assert model.available

    prompt = "alpha beta gamma delta"
    scores = model.score_tokens(prompt)

    # The dummy model should have been invoked once with the full prompt.
    assert len(dummy_model.forward_inputs) == 1
    assert len(scores) == 4

    base = torch.linspace(0.0, 1.0, steps=dummy_model.vocab_size)
    log_probs = entropy.F.log_softmax(base, dim=0)
    probs = log_probs.exp()
    token_ids = [dummy_tokenizer.vocab[token] for token in prompt.split()]
    nll_values = [-log_probs[token_id].item() for token_id in token_ids[1:]]
    fallback_nll = max(nll_values) if nll_values else 0.0

    for index, score in enumerate(scores):
        token_text = prompt[score.start : score.end]
        token_id = dummy_tokenizer.vocab[token_text]
        expected_nll = (
            fallback_nll if index == 0 else float(-log_probs[token_id].item())
        )
        assert pytest.approx(score.entropy, rel=1e-5) == expected_nll
        if index == 0:
            assert score.confidence is None
        else:
            assert pytest.approx(score.confidence or 0.0, rel=1e-5) == float(
                probs[token_id].item()
            )


@pytest.mark.skipif(
    entropy.torch is None or entropy.F is None, reason="torch not available"
)
def test_entropy_scoring_raises_when_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure entropy scoring fails fast when inputs exceed tokenizer limits."""

    torch = entropy.torch
    assert torch is not None  # for mypy

    class OverflowingTokenizer:
        def __init__(self) -> None:
            self.all_special_ids = {0, 1}
            self.model_max_length = 5

        def __call__(
            self,
            text: str,
            return_offsets_mapping: bool = True,
            return_tensors: str = "pt",
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_overflowing_tokens: bool = False,
        ):
            tokens = text.split()
            limit = max_length if truncation and max_length else len(tokens)
            truncated = tokens[:limit]

            offsets = []
            cursor = 0
            for token in truncated:
                start = text.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end + 1

            ids = torch.arange(len(truncated), dtype=torch.long)
            attention = torch.ones_like(ids)
            mapping = torch.tensor(offsets, dtype=torch.long)

            encoded = {
                "input_ids": ids.unsqueeze(0),
                "attention_mask": attention.unsqueeze(0),
                "offset_mapping": mapping.unsqueeze(0),
            }

            if return_overflowing_tokens:
                encoded["overflowing_tokens"] = tokens[limit:]
                encoded["num_truncated_tokens"] = max(0, len(tokens) - len(truncated))

            return encoded

    class NeverCalledModel(torch.nn.Module):
        def forward(self, *args, **kwargs):  # type: ignore[override]
            raise AssertionError(
                "Entropy model should not be invoked for truncated inputs"
            )

    monkeypatch.setattr(entropy, "_SCORER", None)
    monkeypatch.setattr(
        entropy,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: OverflowingTokenizer()),
    )
    monkeypatch.setattr(
        entropy,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: NeverCalledModel()),
    )
    monkeypatch.setattr(
        entropy,
        "resolve_cached_model_path",
        lambda *_args, **_kwargs: "fake-model-path",
    )

    model = entropy._EntropyModel()
    assert model.available

    prompt = " ".join([f"token{i}" for i in range(12)])
    with pytest.raises(RuntimeError) as exc_info:
        model.score_tokens(prompt)
    assert "exceeds tokenizer max length" in str(exc_info.value)


