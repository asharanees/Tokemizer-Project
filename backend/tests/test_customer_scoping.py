from __future__ import annotations

import importlib
import sqlite3
import time
from pathlib import Path

import pytest


def _reload_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "app.db"))
    import database

    importlib.reload(database)
    database.init_db()
    return database


def test_history_customer_scope_contextvar(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = _reload_database(tmp_path, monkeypatch)
    monkeypatch.setenv("OPTIMIZATION_HISTORY_ENABLED", "1")
    importlib.reload(database)
    database.init_db()

    with database.customer_scope("cust_x"):
        record_id = database.record_optimization_history(
            mode="basic",
            raw_prompt="hello",
            optimized_prompt="hi",
            raw_tokens=10,
            optimized_tokens=5,
            processing_time_ms=1.0,
            estimated_cost_before=0.0,
            estimated_cost_after=0.0,
            estimated_cost_saved=0.0,
        )

    assert record_id is not None

    target_db = Path(tmp_path) / "app.db"
    row = None
    deadline = time.time() + 6.0
    while time.time() < deadline and row is None:
        with sqlite3.connect(target_db) as connection:
            row = connection.execute(
                "SELECT customer_id FROM optimization_history WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            time.sleep(0.1)
    assert row[0] == "cust_x"


def test_list_recent_telemetry_is_customer_scoped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = _reload_database(tmp_path, monkeypatch)

    history_id = "opt_1"
    with database.get_db() as conn:
        conn.execute(
            """
            INSERT INTO optimization_history (
                id, customer_id, created_at, updated_at, mode,
                raw_prompt, optimized_prompt, raw_tokens, optimized_tokens,
                processing_time_ms, estimated_cost_before, estimated_cost_after, estimated_cost_saved,
                compression_percentage, semantic_similarity, techniques_applied
            ) VALUES (?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                history_id,
                "cust_a",
                "basic",
                "raw",
                "opt",
                10,
                5,
                1.0,
                0.0,
                0.0,
                0.0,
                50.0,
                None,
                "[]",
            ),
        )
        conn.execute(
            """
            INSERT INTO performance_telemetry (
                optimization_id, pass_name, pass_order, duration_ms,
                tokens_before, tokens_after, tokens_saved, reduction_percent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (history_id, "pass", 1, 1.0, 10, 5, 5, 50.0),
        )
        conn.execute(
            """
            INSERT INTO optimization_history (
                id, customer_id, created_at, updated_at, mode,
                raw_prompt, optimized_prompt, raw_tokens, optimized_tokens,
                processing_time_ms, estimated_cost_before, estimated_cost_after, estimated_cost_saved,
                compression_percentage, semantic_similarity, techniques_applied
            ) VALUES (?, ?, datetime('now'), datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "opt_2",
                "cust_b",
                "basic",
                "raw",
                "opt",
                10,
                5,
                1.0,
                0.0,
                0.0,
                0.0,
                50.0,
                None,
                "[]",
            ),
        )
        conn.execute(
            """
            INSERT INTO performance_telemetry (
                optimization_id, pass_name, pass_order, duration_ms,
                tokens_before, tokens_after, tokens_saved, reduction_percent, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            ("opt_2", "pass", 1, 1.0, 10, 5, 5, 50.0),
        )

    rows_a = database.list_recent_telemetry(10, customer_id="cust_a")
    rows_b = database.list_recent_telemetry(10, customer_id="cust_b")
    rows_none = database.list_recent_telemetry(10, customer_id=None)

    assert {row["optimization_id"] for row in rows_a} == {"opt_1"}
    assert {row["optimization_id"] for row in rows_b} == {"opt_2"}
    assert rows_none == []


def test_batch_jobs_are_customer_scoped(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    database = _reload_database(tmp_path, monkeypatch)
    job_a = database.create_batch_job("A", total_items=1, customer_id="cust_a")
    job_b = database.create_batch_job("B", total_items=1, customer_id="cust_b")

    jobs_a = database.list_batch_jobs(10, customer_id="cust_a")
    jobs_b = database.list_batch_jobs(10, customer_id="cust_b")

    assert [job.id for job in jobs_a] == [job_a.id]
    assert [job.id for job in jobs_b] == [job_b.id]

