import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

app_root = Path(__file__).resolve().parents[1]
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

if "DB_PATH" not in os.environ:
    # Build-time runs use a deterministic temp DB seeded by init_db() for reproducible defaults.
    os.environ["DB_PATH"] = os.path.join(
        tempfile.gettempdir(),
        "tokemizer_pre_download.db",
    )

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("pre_download")


def _load_database() -> Callable[[], None]:
    from database import init_db

    return init_db


def _load_model_cache_manager() -> tuple[
    Callable[..., object], Callable[..., object], Callable[..., object]
]:
    from services.model_cache_manager import (ensure_models_cached,
                                              ensure_spacy_model_cached,
                                              get_model_configs)

    return ensure_models_cached, get_model_configs, ensure_spacy_model_cached


def _resolve_model_types() -> list[str]:
    """Seed a temporary build DB to ensure default model inventory exists."""
    init_db = _load_database()
    _, get_model_configs, _ = _load_model_cache_manager()
    init_db()
    configs = get_model_configs()
    if configs:
        return list(configs.keys())
    return []


def main() -> None:
    ensure_models_cached, _, ensure_spacy_model_cached = _load_model_cache_manager()
    hf_home = os.environ.get("HF_HOME", "/app/.cache/huggingface")
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    logger.info("Starting model pre-download to %s", hf_home)

    model_types = _resolve_model_types()
    if not model_types:
        logger.error("No models found in model_inventory; nothing to download.")
        sys.exit(1)

    available, missing = ensure_models_cached(
        hf_home, model_types, refresh_mode="download_missing"
    )
    if missing:
        logger.error("Some models failed to download: %s", ", ".join(missing))
        sys.exit(1)

    if not ensure_spacy_model_cached(allow_downloads=True):
        logger.error("spaCy model failed to download")
        sys.exit(1)

    logger.info("All models downloaded successfully: %s", ", ".join(available))


if __name__ == "__main__":
    main()
