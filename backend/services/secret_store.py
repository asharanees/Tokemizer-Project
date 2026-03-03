import base64
import hashlib
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken


def _get_fernet() -> Fernet:
    secret = os.environ.get("SECRET_KEY", "dev_secret_key_change_in_production")
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def encrypt_value(value: str) -> str:
    fernet = _get_fernet()
    token = fernet.encrypt(value.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_value(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    fernet = _get_fernet()
    try:
        decrypted = fernet.decrypt(token.encode("utf-8"))
        return decrypted.decode("utf-8")
    except InvalidToken:
        return None
