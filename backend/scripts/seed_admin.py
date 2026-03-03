import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Tuple

# Suppress pkg_resources deprecation warning from quantulum3
warnings.filterwarnings("ignore", category=UserWarning, module=".*quantulum3.*")

# Add backend to path for imports
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))


def _load_dependencies() -> Tuple[
    Any,
    Callable[..., Any],
    Callable[..., Any],
    str,
    Callable[..., Any],
    Callable[..., Any],
]:
    import auth_utils
    from database import (DB_PATH, create_customer, get_customer_by_email,
                          init_db, update_customer)

    return (
        auth_utils,
        create_customer,
        init_db,
        DB_PATH,
        get_customer_by_email,
        update_customer,
    )


def _running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def _warn_if_container_db_path_on_windows() -> None:
    if os.name != "nt" or _running_in_docker():
        return
    env_db_path = os.environ.get("DB_PATH", "")
    if env_db_path.replace("\\", "/").startswith("/app/"):
        print(f"[error] DB_PATH points to a container path on Windows: {env_db_path}")
        print(
            "[error] Run this script inside the backend container, or unset DB_PATH to use local DB."
        )
        sys.exit(1)


def seed_admin(email: str, password: str, name: str = "Admin") -> None:
    auth_utils, create_customer, init_db, _, get_customer_by_email, update_customer = (
        _load_dependencies()
    )
    init_db()

    existing = get_customer_by_email(email)
    pwd_hash = auth_utils.get_password_hash(password)

    if existing:
        print(f"Updating existing user {email} to admin...")
        updated = update_customer(
            existing.id,
            password_hash=pwd_hash,
            role="admin",
            is_active=True,
            subscription_status="active",
            subscription_tier="enterprise",
            quota_override=9999999,
        )
        if updated and updated.password_hash:
            # Verify the password was correctly stored
            verify_ok = auth_utils.verify_password(password, updated.password_hash)
            if verify_ok:
                print(f"[ok] Admin user {email} updated successfully")
                print("[ok] Password verified successfully")
            else:
                print("[error] Password verification failed!")
        else:
            print(f"[error] Failed to update admin user {email}")
    else:
        print(f"Creating new admin user {email}...")
        admin = create_customer(name=name, email=email)
        updated = update_customer(
            admin.id,
            password_hash=pwd_hash,
            role="admin",
            is_active=True,
            subscription_status="active",
            subscription_tier="enterprise",
            quota_override=9999999,
        )
        if updated and updated.password_hash:
            # Verify the password was correctly stored
            verify_ok = auth_utils.verify_password(password, updated.password_hash)
            if verify_ok:
                print(f"[ok] Admin user {email} created successfully")
                print("[ok] Password verified successfully")
            else:
                print("[error] Password verification failed!")
        else:
            print(f"[error] Failed to create admin user {email}")

    print("Done.")


def _print_usage() -> None:
    print("Usage (from repo root):")
    print("  python backend/scripts/seed_admin.py <email> <password> [name]")
    print()
    print("Usage (inside backend container or from backend dir):")
    print("  python scripts/seed_admin.py <email> <password> [name]")
    print()
    print("Note: If your password contains special characters (&, |, etc.),")
    print(
        "      wrap it in quotes: python scripts/seed_admin.py email 'password&with&special' Name"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        _print_usage()
        sys.exit(1)

    _warn_if_container_db_path_on_windows()

    email = sys.argv[1]
    password = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "Admin"
    _, _, _, DB_PATH, _, _ = _load_dependencies()

    print(f"Email: {email}")
    print(f"Password length: {len(password)}")
    print(f"Name: {name}")
    print(f"DB_PATH: {DB_PATH}")
    print()

    seed_admin(email, password, name)
