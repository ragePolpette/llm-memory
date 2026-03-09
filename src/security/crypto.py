"""Cifratura opzionale payload sensibile (locale)."""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass


class EncryptionConfigurationError(RuntimeError):
    """Configurazione cifratura non valida."""


@dataclass
class CipherResult:
    """Risultato cifratura payload."""

    payload: str
    encrypted: bool


class PayloadCipher:
    """Interfaccia semplice per cifrare/decifrare payload testuale."""

    def encrypt(self, plaintext: str) -> CipherResult:  # pragma: no cover - interface
        raise NotImplementedError

    def decrypt(self, payload: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class NoOpCipher(PayloadCipher):
    """Cipher nullo (default)."""

    def encrypt(self, plaintext: str) -> CipherResult:
        return CipherResult(payload=plaintext, encrypted=False)

    def decrypt(self, payload: str) -> str:
        return payload


class FernetCipher(PayloadCipher):
    """Cipher basato su cryptography.fernet con chiave locale."""

    _PASSPHRASE_MIN_LENGTH = 32
    _KDF_SALT = b"llm-memory::fernet::v1"
    _KDF_ITERATIONS = 200_000

    def __init__(self, key: str):
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:  # pragma: no cover
            raise EncryptionConfigurationError(
                "Per usare MEMORY_ENCRYPTION_ENABLED serve `cryptography` installato localmente."
            ) from exc

        self._fernet = Fernet(self._build_fernet_key(key))

    @classmethod
    def _build_fernet_key(cls, key: str) -> bytes:
        key_bytes = key.encode("utf-8")

        if len(key_bytes) == 44:
            try:
                decoded = base64.urlsafe_b64decode(key_bytes)
            except Exception:
                decoded = b""
            if len(decoded) == 32:
                return key_bytes

        if len(key_bytes) < cls._PASSPHRASE_MIN_LENGTH:
            raise EncryptionConfigurationError(
                "Encryption key must be a valid Fernet key or a passphrase with at least 32 bytes."
            )

        derived = hashlib.pbkdf2_hmac(
            "sha256",
            key_bytes,
            cls._KDF_SALT,
            cls._KDF_ITERATIONS,
            dklen=32,
        )
        return base64.urlsafe_b64encode(derived)

    def encrypt(self, plaintext: str) -> CipherResult:
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return CipherResult(payload=token.decode("utf-8"), encrypted=True)

    def decrypt(self, payload: str) -> str:
        return self._fernet.decrypt(payload.encode("utf-8")).decode("utf-8")


def build_cipher(enabled: bool, key_env: str) -> PayloadCipher:
    """Factory per cifratura opzionale con key locale da env."""

    if not enabled:
        return NoOpCipher()

    key = os.getenv(key_env)
    if not key:
        raise EncryptionConfigurationError(
            f"Cifratura abilitata ma env `{key_env}` non impostata."
        )

    return FernetCipher(key)
