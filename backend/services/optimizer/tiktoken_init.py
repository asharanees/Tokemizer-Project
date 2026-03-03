# Improved tiktoken initialization code with offline support
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def init_tiktoken(tiktoken_module):
    """
    Initialize tiktoken with offline-only support via vendored encodings.

    Returns:
        Tokenizer instance or None if initialization fails
    """
    offline_mode = True

    try:
        # Set cache directory for vendored encodings if not already set
        # This allows offline operation by using pre-downloaded encodings
        if "TIKTOKEN_CACHE_DIR" not in os.environ:
            # Fix: Use pathlib to correctly resolve vendor path
            # backend/services/optimizer/tiktoken_init.py -> backend/vendor/tiktoken
            vendor_path = Path(__file__).resolve().parents[2] / "vendor" / "tiktoken"

            if vendor_path.exists():
                os.environ["TIKTOKEN_CACHE_DIR"] = str(vendor_path)
                logger.info(f"Using vendored tiktoken encodings from {vendor_path}")
            elif offline_mode:
                # In strict offline mode, fail fast with actionable message
                error_msg = (
                    f"❌ No local tiktoken encoding at {vendor_path}. "
                    f'Run: make vendor-tiktoken (or python -c "import tiktoken; '
                    f"import os; os.environ['TIKTOKEN_CACHE_DIR'] = '{vendor_path}'; "
                    f"tiktoken.get_encoding('cl100k_base')\")"
                )
                logger.error(error_msg)
                return None

        os.environ["TIKTOKEN_OFFLINE"] = "1"

        # Load the encoding (will use vendored files if available)
        tokenizer = tiktoken_module.get_encoding("cl100k_base")

        # Log initialization status
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", "default (~/.tiktoken)")
        mode_str = " [OFFLINE MODE]" if offline_mode else ""
        logger.info(
            f"✅ Successfully initialized tiktoken with cl100k_base encoding{mode_str} "
            f"(cache: {cache_dir})"
        )

        # Validate the tokenizer works
        test_tokens = tokenizer.encode("test")
        if not test_tokens:
            raise ValueError("Tokenizer validation failed - empty encoding")

        return tokenizer

    except Exception as e:
        error_msg = f"❌ Failed to initialize tiktoken: {e}."

        # Provide actionable error message
        if not os.environ.get("TIKTOKEN_CACHE_DIR"):
            vendor_path = Path(__file__).resolve().parents[2] / "vendor" / "tiktoken"
            error_msg += (
                f" No local tiktoken encoding at {vendor_path}. "
                f"Run: make vendor-tiktoken"
            )
        else:
            error_msg += " Token counting will use fallback estimation method with reduced accuracy."

        logger.error(error_msg)
        return None
