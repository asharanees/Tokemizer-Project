#!/usr/bin/env python3
"""Check current model inventory."""

import logging
import sqlite3
import sys
from pathlib import Path

# Setup path to find database module
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from database import DB_PATH, init_db  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def check_models():
    try:
        logger.info(f"Using database at: {DB_PATH}")

        # Ensure DB is initialized (will create tables if missing, but won't overwrite if exists)
        init_db()

        # Connect directly to verify content
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(
            """
            SELECT model_type, model_name, min_size_bytes, expected_files, revision, allow_patterns
            FROM model_inventory
            """
        )
        rows = cursor.fetchall()

        if not rows:
            logger.warning("No models found in inventory!")
            return

        logger.info(f"Found {len(rows)} models in inventory:")
        for row in rows:
            logger.info("-" * 40)
            logger.info(f"Type: {row['model_type']}")
            logger.info(f"Name: {row['model_name']}")
            logger.info(f"Min Size: {row['min_size_bytes']} bytes")
            logger.info(f"Files: {row['expected_files']}")
            logger.info(f"Revision: {row['revision']}")
            logger.info(f"Allow Patterns: {row['allow_patterns']}")

        conn.close()

    except Exception as e:
        logger.error(f"Error checking models: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    check_models()
