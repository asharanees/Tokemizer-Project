import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from database import get_db


def get_model_inventory_item(model_type: str) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM model_inventory WHERE model_type = ? AND is_active = 1",
            (model_type,),
        ).fetchone()
        if not row:
            return None
        return dict(row)


def list_model_inventory() -> List[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM model_inventory WHERE is_active = 1 ORDER BY model_type"
        ).fetchall()
        return [dict(row) for row in rows]


def add_or_update_model_inventory(
    model_type: str,
    model_name: str,
    min_size_bytes: int,
    expected_files: List[str],
    component: Optional[str] = None,
    library_type: Optional[str] = None,
    usage: Optional[str] = None,
    revision: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        # Check if exists (active or inactive)
        row = conn.execute(
            "SELECT id FROM model_inventory WHERE model_type = ?", (model_type,)
        ).fetchone()

        expected_files_json = json.dumps(expected_files)
        component_value = (component or "").strip()
        library_value = (library_type or "").strip()
        usage_value = (usage or "").strip()
        revision_value = (revision or "").strip()
        allow_patterns_json = json.dumps(allow_patterns or [])

        if row:
            # Update
            conn.execute(
                """
                UPDATE model_inventory
                SET model_name = ?, min_size_bytes = ?, expected_files = ?, component = ?, library_type = ?,
                    usage = ?, revision = ?, allow_patterns = ?, is_active = 1, updated_at = ?
                WHERE model_type = ?
            """,
                (
                    model_name,
                    min_size_bytes,
                    expected_files_json,
                    component_value,
                    library_value,
                    usage_value,
                    revision_value,
                    allow_patterns_json,
                    timestamp,
                    model_type,
                ),
            )
            record_id = row["id"]
        else:
            # Insert
            record_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO model_inventory (
                    id, model_type, model_name, min_size_bytes, expected_files, component,
                    library_type, usage, revision, allow_patterns, is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
                (
                    record_id,
                    model_type,
                    model_name,
                    min_size_bytes,
                    expected_files_json,
                    component_value,
                    library_value,
                    usage_value,
                    revision_value,
                    allow_patterns_json,
                    timestamp,
                    timestamp,
                ),
            )

    return get_model_inventory_item(model_type)


def delete_model_inventory(model_type: str) -> bool:
    # Soft delete
    timestamp = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cursor = conn.execute(
            """
            UPDATE model_inventory
            SET is_active = 0, updated_at = ?
            WHERE model_type = ? AND is_active = 1
        """,
            (timestamp, model_type),
        )
        return cursor.rowcount > 0
