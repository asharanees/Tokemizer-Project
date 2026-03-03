from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "seed_admin.py"
)


def _run_seed_admin(db_path: Path, email: str, password: str, name: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DB_PATH"] = str(db_path)
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), email, password, name],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_seed_admin_creates_and_updates_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    email = f"seed-{int(time.time() * 1000)}@example.com"
    password = "StrongPass!123"
    name = "Seed Admin"

    first = _run_seed_admin(db_path, email, password, name)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "Admin user" in first.stdout
    assert "Password verified successfully" in first.stdout

    second = _run_seed_admin(db_path, email, password, name)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Updating existing user" in second.stdout
    assert "Password verified successfully" in second.stdout

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            """
            SELECT email, role, is_active, subscription_status, subscription_tier, quota_override
            FROM customers WHERE email = ?
            """,
            (email,),
        ).fetchone()

    assert row is not None
    assert row[0] == email
    assert row[1] == "admin"
    assert row[2] == 1
    assert row[3] == "active"
    assert row[4] == "enterprise"
    assert row[5] == 9999999


def test_seed_admin_windows_host_guard_for_container_db_path(tmp_path: Path) -> None:
    if os.name != "nt":
        return

    env = os.environ.copy()
    env["DB_PATH"] = "/app/data/app.db"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "admin@example.com", "Pass123!x", "Admin"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert proc.returncode == 1
    assert "DB_PATH points to a container path on Windows" in proc.stdout
