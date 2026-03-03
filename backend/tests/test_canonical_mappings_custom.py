from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _reload_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    import database

    importlib.reload(database)
    database.init_db()
    return database


def test_custom_canonical_mapping_overrides_ootb_case_insensitive(tmp_path, monkeypatch):
    database = _reload_database(tmp_path, monkeypatch)

    with database.get_db() as conn:
        conn.execute(
            """
            INSERT INTO canonical_mappings (source_token, target_token, created_at, updated_at)
            VALUES (lower(?), ?, datetime('now'), datetime('now'))
            """,
            ("Some Term", "OOTB",),
        )

    database.create_user_canonical_mapping("cust_a", "Some Term", "CUSTOM")

    combined = database.get_combined_canonical_mappings("cust_a")
    assert combined["some term"] == "CUSTOM"


def test_disabled_ootb_does_not_remove_custom_override(tmp_path, monkeypatch):
    database = _reload_database(tmp_path, monkeypatch)

    with database.get_db() as conn:
        conn.execute(
            """
            INSERT INTO canonical_mappings (source_token, target_token, created_at, updated_at)
            VALUES (lower(?), ?, datetime('now'), datetime('now'))
            """,
            ("Disable Me", "OOTB",),
        )

    database.toggle_ootb_mapping("cust_a", "Disable Me", enabled=False)
    database.create_user_canonical_mapping("cust_a", "Disable Me", "CUSTOM")

    combined = database.get_combined_canonical_mappings("cust_a")
    assert combined["disable me"] == "CUSTOM"


def test_custom_canonical_mappings_are_customer_scoped(tmp_path, monkeypatch):
    database = _reload_database(tmp_path, monkeypatch)

    database.create_user_canonical_mapping("cust_a", "shared", "A")
    database.create_user_canonical_mapping("cust_b", "shared", "B")

    combined_a = database.get_combined_canonical_mappings("cust_a")
    combined_b = database.get_combined_canonical_mappings("cust_b")

    assert combined_a["shared"] == "A"
    assert combined_b["shared"] == "B"
