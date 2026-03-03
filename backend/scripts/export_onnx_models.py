import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

logger = logging.getLogger(__name__)


def _load_model_cache_manager() -> Tuple[
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
    Callable[..., Any],
]:
    from services.model_cache_manager import (ensure_models_cached,
                                              get_model_configs,
                                              resolve_cached_model_artifact,
                                              resolve_cached_model_path)

    return (
        ensure_models_cached,
        get_model_configs,
        resolve_cached_model_artifact,
        resolve_cached_model_path,
    )


def _resolve_hf_home() -> str:
    return os.environ.get("HF_HOME", "/app/.cache/huggingface")


def _load_torch_modules():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import (AutoModel, AutoModelForTokenClassification,
                                  AutoTokenizer)
    except ImportError:
        return None
    return torch, F, AutoModel, AutoModelForTokenClassification, AutoTokenizer


def _supports_token_type_ids(model) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "token_type_ids" in signature.parameters


def _prepare_inputs(tokenizer, include_token_type_ids: bool):
    encoded = tokenizer("Export ONNX", return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)
    if include_token_type_ids:
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = input_ids.new_zeros(input_ids.shape)
        return input_ids, attention_mask, token_type_ids
    return input_ids, attention_mask


def _export_with_quantization(model_path: str, output_path: str) -> Optional[str]:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        logger.warning("onnxruntime quantization unavailable; skipping int8 export")
        return None
    quantized_path = os.path.join(model_path, "model.int8.onnx")
    quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QInt8)
    return quantized_path


def _save_tokenizer_and_config(tokenizer, model, output_dir: str) -> None:
    tokenizer.save_pretrained(output_dir)
    model.config.to_json_file(os.path.join(output_dir, "config.json"))


def _infer_model_type_from_architectures(
    architectures: Optional[List[str]],
) -> Optional[str]:
    if not architectures:
        return None
    arch = architectures[0].lower()
    mapping = (
        ("distilbert", "distilbert"),
        ("mobilebert", "mobilebert"),
        ("roberta", "roberta"),
        ("xlm-roberta", "xlm-roberta"),
        ("xlm", "xlm"),
        ("albert", "albert"),
        ("electra", "electra"),
        ("funnel", "funnel"),
        ("bart", "bart"),
        ("mbart", "mbart"),
        ("gpt2", "gpt2"),
        ("bert", "bert"),
    )
    for needle, model_type in mapping:
        if needle in arch:
            return model_type
    return None


def _ensure_config_has_model_type(config_path: Optional[str], model_name: str) -> None:
    if not config_path or not os.path.isfile(config_path):
        return
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except json.JSONDecodeError:
        logger.warning("Unable to read config.json for %s", model_name)
        return

    if "model_type" in config:
        return

    inferred = _infer_model_type_from_architectures(config.get("architectures"))
    config["model_type"] = inferred or "bert"

    try:
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
    except OSError as exc:
        logger.warning("Unable to persist updated config for %s: %s", model_name, exc)


def _config_looks_like_transformer(config_path: str) -> bool:
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
    except json.JSONDecodeError:
        return False
    return bool(config.get("architectures") or config.get("model_type"))


def _resolve_sentence_transformer_path(
    model_type: str, model_name: str, model_path: str
) -> str:
    _, _, resolve_cached_model_artifact, _ = _load_model_cache_manager()
    modules_path = resolve_cached_model_artifact(model_type, model_name, "modules.json")
    if modules_path and os.path.isfile(modules_path):
        try:
            with open(modules_path, "r", encoding="utf-8") as fh:
                modules = json.load(fh)
        except json.JSONDecodeError:
            logger.warning("Unable to read modules.json in %s", modules_path)
            return os.path.dirname(modules_path)
        base_dir = os.path.dirname(modules_path)
        zero_transformer = os.path.join(base_dir, "0_Transformer")
        if os.path.isdir(zero_transformer):
            return zero_transformer
        for module in modules:
            module_type = str(module.get("type", "")).lower()
            leaf = module_type.split(".")[-1]
            is_transformer_module = (
                leaf == "transformer" or ".models.transformer" in module_type
            )
            if not is_transformer_module:
                continue
            module_path = module.get("path")
            if not module_path:
                continue
            candidate = os.path.join(base_dir, module_path)
            if os.path.isdir(candidate):
                return candidate

    config_path = resolve_cached_model_artifact(model_type, model_name, "config.json")
    tokenizer_path = resolve_cached_model_artifact(
        model_type, model_name, "tokenizer.json"
    )
    if config_path and tokenizer_path:
        config_dir = os.path.dirname(config_path)
        if config_dir == os.path.dirname(tokenizer_path):
            return config_dir

    for root, _dirs, files in os.walk(model_path):
        if "config.json" not in files:
            continue
        candidate_config = os.path.join(root, "config.json")
        if not _config_looks_like_transformer(candidate_config):
            continue
        if (
            "tokenizer.json" in files
            or "vocab.txt" in files
            or "tokenizer_config.json" in files
        ):
            return root

    snapshots_path = os.path.join(model_path, "snapshots")
    if os.path.isdir(snapshots_path):
        for snapshot in os.listdir(snapshots_path):
            snapshot_dir = os.path.join(snapshots_path, snapshot)
            if not os.path.isdir(snapshot_dir):
                continue
            for root, _dirs, files in os.walk(snapshot_dir):
                if "config.json" not in files:
                    continue
                candidate_config = os.path.join(root, "config.json")
                if not _config_looks_like_transformer(candidate_config):
                    continue
                if (
                    "tokenizer.json" in files
                    or "vocab.txt" in files
                    or "tokenizer_config.json" in files
                ):
                    return root

    return model_path


def export_semantic_guard(
    model_name: str, model_path: str, *, model_type: str = "semantic_guard"
) -> bool:
    modules = _load_torch_modules()
    if modules is None:
        logger.error("Missing torch or transformers; cannot export semantic guard")
        return False
    torch, F, AutoModel, _, AutoTokenizer = modules
    transformer_path = _resolve_sentence_transformer_path(
        model_type,
        model_name,
        model_path,
    )
    logger.info(
        "Resolved %s transformer_path: %s", model_type, transformer_path
    )
    config_path = os.path.join(transformer_path, "config.json")
    _ensure_config_has_model_type(config_path, model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            transformer_path, local_files_only=True
        )
        base_model = AutoModel.from_pretrained(transformer_path, local_files_only=True)
    except Exception as exc:
        logger.error("Failed to load semantic guard model %s: %s", model_name, exc)
        return False

    include_token_type_ids = _supports_token_type_ids(base_model)

    class SemanticGuardWrapper(torch.nn.Module):
        def __init__(self, model, use_token_type_ids: bool) -> None:
            super().__init__()
            self.model = model
            self.use_token_type_ids = use_token_type_ids

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            if self.use_token_type_ids and token_type_ids is not None:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs[0]
            mask = attention_mask.unsqueeze(-1).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            return F.normalize(pooled, p=2, dim=1)

    wrapper = SemanticGuardWrapper(base_model, include_token_type_ids).eval()
    output_path = os.path.join(transformer_path, "model.onnx")

    if include_token_type_ids:
        input_ids, attention_mask, token_type_ids = _prepare_inputs(tokenizer, True)
        inputs = (input_ids, attention_mask, token_type_ids)
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "embeddings": {0: "batch"},
        }
    else:
        input_ids, attention_mask = _prepare_inputs(tokenizer, False)
        inputs = (input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "embeddings": {0: "batch"},
        }

    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=input_names,
        output_names=["embeddings"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    _save_tokenizer_and_config(tokenizer, base_model, transformer_path)
    _export_with_quantization(transformer_path, output_path)
    return True


def export_token_classifier(model_name: str, model_path: str) -> bool:
    modules = _load_torch_modules()
    if modules is None:
        logger.error("Missing torch or transformers; cannot export token classifier")
        return False
    torch, _, _, AutoModelForTokenClassification, AutoTokenizer = modules
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        base_model = AutoModelForTokenClassification.from_pretrained(
            model_path, local_files_only=True
        )
    except Exception as exc:
        logger.error("Failed to load token classifier model %s: %s", model_name, exc)
        return False

    include_token_type_ids = _supports_token_type_ids(base_model)

    class TokenClassifierWrapper(torch.nn.Module):
        def __init__(self, model, use_token_type_ids: bool) -> None:
            super().__init__()
            self.model = model
            self.use_token_type_ids = use_token_type_ids

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            if self.use_token_type_ids and token_type_ids is not None:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    wrapper = TokenClassifierWrapper(base_model, include_token_type_ids).eval()
    output_path = os.path.join(model_path, "model.onnx")

    if include_token_type_ids:
        input_ids, attention_mask, token_type_ids = _prepare_inputs(tokenizer, True)
        inputs = (input_ids, attention_mask, token_type_ids)
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        }
    else:
        input_ids, attention_mask = _prepare_inputs(tokenizer, False)
        inputs = (input_ids, attention_mask)
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        }

    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    _save_tokenizer_and_config(tokenizer, base_model, model_path)
    _export_with_quantization(model_path, output_path)
    return True


def _resolve_model_paths(model_types: List[str]) -> List[Tuple[str, str, str]]:
    _, get_model_configs, _, resolve_cached_model_path = _load_model_cache_manager()
    configs = get_model_configs()
    resolved: List[Tuple[str, str, str]] = []
    for model_type in model_types:
        entry = configs.get(model_type)
        if not entry or not entry.get("model_name"):
            logger.error("Missing model entry for %s", model_type)
            continue
        model_name = entry["model_name"]
        model_path = resolve_cached_model_path(model_type, model_name)
        if model_path is None:
            logger.error("Model cache path not found for %s", model_name)
            continue
        resolved.append((model_type, model_name, model_path))
    return resolved


def _resolve_candidate_models() -> List[Tuple[str, str, str]]:
    _, _, _, resolve_cached_model_path = _load_model_cache_manager()
    raw = os.environ.get("PROMPT_OPTIMIZER_TOKEN_CLASSIFIER_CANDIDATES", "")
    if not raw.strip():
        return []
    candidates = [item.strip() for item in raw.split(",") if item.strip()]
    resolved: List[Tuple[str, str, str]] = []
    for name in candidates:
        model_path = resolve_cached_model_path("token_classifier", name)
        if model_path is None:
            logger.warning("Candidate model cache path not found for %s", name)
            continue
        resolved.append(("token_classifier", name, model_path))
    return resolved


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    hf_home = _resolve_hf_home()
    ensure_models_cached, _, _, _ = _load_model_cache_manager()
    ensure_models_cached(
        hf_home,
        ["semantic_guard", "token_classifier"],
        refresh_mode="download_missing",
    )
    resolved = _resolve_model_paths(["semantic_guard", "token_classifier"])
    resolved.extend(_resolve_candidate_models())
    deduped: List[Tuple[str, str, str]] = []
    seen = set()
    for entry in resolved:
        key = (entry[0], entry[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    resolved = deduped
    if not resolved:
        logger.error("No models resolved for ONNX export")
        return 1

    status = 0
    for model_type, model_name, model_path in resolved:
        if model_type == "token_classifier":
            ok = export_token_classifier(model_name, model_path)
        else:
            ok = export_semantic_guard(model_name, model_path)
        if not ok:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
