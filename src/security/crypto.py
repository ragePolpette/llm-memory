"""Cifratura opzionale payload sensibile (locale)."""

from __future__ import annotations

import base64
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

    def __init__(self, key: str):
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:  # pragma: no cover
            raise EncryptionConfigurationError(
                "Per usare MEMORY_ENCRYPTION_ENABLED serve `cryptography` installato localmente."
            ) from exc

        key_bytes = key.encode("utf-8")
        # Supporta chiavi raw convertibili in base64-url-safe.
        if len(key_bytes) != 44:
            key_bytes = base64.urlsafe_b64encode(key_bytes.ljust(32, b"0")[:32])
        self._fernet = Fernet(key_bytes)

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
