"""Unit tests for token classifier compression helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
from services.optimizer import token_classifier
from services.optimizer.core import PromptOptimizer
from services.optimizer.router import get_profile


def test_token_classifier_model_unavailable_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure token classifier fails fast when model is unavailable."""
    monkeypatch.setattr(token_classifier, "AutoTokenizer", None)
    monkeypatch.setattr(token_classifier, "ort", None)

    monkeypatch.setattr(token_classifier, "_CLASSIFIER", None)

    text = "This is a test prompt that should pass through unchanged."
    with pytest.raises(RuntimeError) as exc_info:
        token_classifier.compress_with_token_classifier(text)
    assert "Token classifier is unavailable" in str(exc_info.value)


@pytest.mark.skipif(
    token_classifier.np is None,
    reason="numpy not available",
)
def test_token_classifier_with_protected_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure token classifier respects protected ranges."""
    np = token_classifier.np
    assert np is not None

    class DummyTokenizer:
        def __init__(self) -> None:
            self.all_special_ids = {0, 1}
            self.model_max_length = 512

        def __call__(
            self,
            text: str,
            return_offsets_mapping: bool = True,
            return_tensors: str = "pt",
            truncation: bool = False,
            max_length: Optional[int] = None,
            stride: Optional[int] = None,
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

            ids = np.arange(len(tokens), dtype=np.int64)
            attention = np.ones_like(ids)
            mapping = np.array(offsets, dtype=np.int64)
            return {
                "input_ids": ids.reshape(1, -1),
                "attention_mask": attention.reshape(1, -1),
                "offset_mapping": mapping.reshape(1, -1, 2),
            }

    class DummySession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def run(self, _outputs, inputs):
            input_ids = inputs["input_ids"]
            batch, seq = input_ids.shape
            logits = np.zeros((batch, seq, 2), dtype=np.float32)
            for i in range(seq):
                if i % 2 == 0:
                    logits[0, i, 1] = 10.0
                else:
                    logits[0, i, 0] = 10.0
            return [logits]

    monkeypatch.setattr(
        token_classifier,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        token_classifier,
        "ort",
        SimpleNamespace(InferenceSession=lambda *_args, **_kwargs: DummySession()),
    )
    monkeypatch.setattr(
        token_classifier,
        "resolve_cached_model_artifact",
        lambda *_args, **_kwargs: "fake.onnx",
    )
    monkeypatch.setattr(token_classifier, "_CLASSIFIER", None)
    monkeypatch.setattr(
        token_classifier,
        "_entropy_high_ranges",
        lambda *_args, **_kwargs: [],
    )

    text = "alpha beta gamma delta"
    # Protect "beta" (token at position 6-10)
    protected_ranges = [(6, 10)]

    result, applied, _metadata = token_classifier.compress_with_token_classifier(
        text, protected_ranges=protected_ranges
    )

    assert applied is True
    # "beta" should be kept because it's protected
    assert "beta" in result
    # Protected content is always preserved
    # Other tokens may or may not be kept depending on classifier


def test_compress_with_token_classifier_no_model() -> None:
    """Token classifier must fail fast when model is unavailable."""
    text = "Sample text for compression test."
    with pytest.raises(RuntimeError) as exc_info:
        token_classifier.compress_with_token_classifier(
            text, model_name="nonexistent/model"
        )
    assert "Token classifier is unavailable" in str(exc_info.value)


def test_token_classifier_warm_up_unavailable() -> None:
    """Test warm_up function when model is unavailable."""
    # Should not raise an exception
    token_classifier.warm_up(model_name="nonexistent/model")


@pytest.mark.skipif(
    token_classifier.np is None,
    reason="numpy not available",
)
def test_token_classifier_empty_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that empty decisions result in no compression."""
    np = token_classifier.np
    assert np is not None

    class DummyTokenizer:
        def __init__(self) -> None:
            self.all_special_ids = {0, 1, 2, 3}  # Mark all as special
            self.model_max_length = 512

        def __call__(
            self,
            text: str,
            return_offsets_mapping: bool = True,
            return_tensors: str = "pt",
            truncation: bool = False,
            max_length: Optional[int] = None,
            stride: Optional[int] = None,
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

            ids = np.arange(len(tokens), dtype=np.int64)
            attention = np.ones_like(ids)
            mapping = np.array(offsets, dtype=np.int64)
            return {
                "input_ids": ids.reshape(1, -1),
                "attention_mask": attention.reshape(1, -1),
                "offset_mapping": mapping.reshape(1, -1, 2),
            }

    class DummySession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def run(self, _outputs, inputs):
            input_ids = inputs["input_ids"]
            batch, seq = input_ids.shape
            logits = np.zeros((batch, seq, 2), dtype=np.float32)
            logits[:, :, 0] = 10.0
            return [logits]

    monkeypatch.setattr(
        token_classifier,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        token_classifier,
        "ort",
        SimpleNamespace(InferenceSession=lambda *_args, **_kwargs: DummySession()),
    )
    monkeypatch.setattr(
        token_classifier,
        "resolve_cached_model_artifact",
        lambda *_args, **_kwargs: "fake.onnx",
    )
    monkeypatch.setattr(token_classifier, "_CLASSIFIER", None)
    monkeypatch.setattr(
        token_classifier,
        "_entropy_high_ranges",
        lambda *_args, **_kwargs: [],
    )

    text = "alpha beta gamma"
    result, applied, _metadata = token_classifier.compress_with_token_classifier(text)

    # All tokens are marked special; classifier should handle this edge case without error.
    assert isinstance(result, str)
    if result == text:
        assert applied is False
    else:
        assert result == ""
        assert applied is True


@pytest.mark.skipif(
    token_classifier.np is None,
    reason="numpy not available",
)
def test_token_classifier_respects_min_keep_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = token_classifier.np
    assert np is not None

    class DummyTokenizer:
        def __init__(self) -> None:
            self.all_special_ids = {0, 1}
            self.model_max_length = 512

        def __call__(
            self,
            text: str,
            return_offsets_mapping: bool = True,
            return_tensors: str = "pt",
            truncation: bool = False,
            max_length: Optional[int] = None,
            stride: Optional[int] = None,
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

            ids = np.arange(len(tokens), dtype=np.int64)
            attention = np.ones_like(ids)
            mapping = np.array(offsets, dtype=np.int64)
            return {
                "input_ids": ids.reshape(1, -1),
                "attention_mask": attention.reshape(1, -1),
                "offset_mapping": mapping.reshape(1, -1, 2),
            }

    class DummySession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def run(self, _outputs, inputs):
            input_ids = inputs["input_ids"]
            batch, seq = input_ids.shape
            logits = np.zeros((batch, seq, 2), dtype=np.float32)
            for i in range(seq):
                if i % 2 == 0:
                    logits[0, i, 1] = 10.0
                else:
                    logits[0, i, 0] = 10.0
            return [logits]

    monkeypatch.setattr(
        token_classifier,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: DummyTokenizer()),
    )
    monkeypatch.setattr(
        token_classifier,
        "ort",
        SimpleNamespace(InferenceSession=lambda *_args, **_kwargs: DummySession()),
    )
    monkeypatch.setattr(
        token_classifier,
        "resolve_cached_model_artifact",
        lambda *_args, **_kwargs: "fake.onnx",
    )
    monkeypatch.setattr(token_classifier, "_CLASSIFIER", None)
    monkeypatch.setattr(
        token_classifier,
        "_entropy_high_ranges",
        lambda *_args, **_kwargs: [],
    )

    text = "alpha beta gamma delta"
    result, applied, _metadata = token_classifier.compress_with_token_classifier(
        text, min_keep_ratio=0.75
    )

    assert result == text
    assert applied is False


@pytest.mark.parametrize("profile_name", ["code", "dialogue"])
def test_token_classifier_uses_profile_thresholds(
    monkeypatch: pytest.MonkeyPatch, profile_name: str
) -> None:
    optimizer = PromptOptimizer()
    captured: dict[str, float] = {}

    def fake_compress(
        text: str,
        *,
        protected_ranges=None,
        model_name=None,
        min_confidence=None,
        min_keep_ratio=None,
        content_profile_name=None,
    ):
        captured["min_confidence"] = min_confidence
        captured["min_keep_ratio"] = min_keep_ratio
        captured["content_profile_name"] = content_profile_name
        return text, False, {}

    monkeypatch.setattr(
        token_classifier, "compress_with_token_classifier", fake_compress
    )

    profile = get_profile(profile_name)
    optimizer._optimize_with_token_classifier(
        "alpha beta gamma",
        force_preserve_digits=True,
        content_type=profile_name,
        content_profile=profile,
    )

    assert (
        captured["min_confidence"]
        == profile.smart_defaults["classifier_min_confidence"]
    )
    assert (
        captured["min_keep_ratio"]
        == profile.smart_defaults["classifier_min_keep_ratio"]
    )
    assert captured["content_profile_name"] == profile.name
