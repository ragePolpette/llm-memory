"""Optional local encryption for sensitive payloads."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class EncryptionConfigurationError(RuntimeError):
    """Invalid encryption configuration."""


class PayloadDecryptionError(RuntimeError):
    """Encrypted payload cannot be decrypted with the current cipher."""


@dataclass
class CipherResult:
    """Payload encryption result."""

    payload: str
    encrypted: bool


class PayloadCipher:
    """Minimal interface to encrypt/decrypt text payloads."""

    def encrypt(self, plaintext: str) -> CipherResult:  # pragma: no cover - interface
        raise NotImplementedError

    def decrypt(self, payload: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class NoOpCipher(PayloadCipher):
    """Pass-through cipher (default)."""

    def encrypt(self, plaintext: str) -> CipherResult:
        return CipherResult(payload=plaintext, encrypted=False)

    def decrypt(self, payload: str) -> str:
        return payload


SALT_BYTES = 16


def load_or_create_salt(salt_path: Path) -> bytes:
    """Return the per-installation KDF salt, creating it on first use."""

    path = Path(salt_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="ascii") as handle:
                handle.write(os.urandom(SALT_BYTES).hex())
    try:
        salt = bytes.fromhex(path.read_text(encoding="ascii").strip())
    except ValueError as exc:
        raise EncryptionConfigurationError(f"Encryption salt file '{path}' is corrupted.") from exc
    if len(salt) < SALT_BYTES:
        raise EncryptionConfigurationError(f"Encryption salt file '{path}' is too short.")
    return salt


class FernetCipher(PayloadCipher):
    """Cipher based on cryptography.fernet with a local key."""

    _PASSPHRASE_MIN_LENGTH = 32
    _LEGACY_KDF_SALT = b"llm-memory::fernet::v1"
    _KDF_ITERATIONS = 200_000

    def __init__(self, key: str, salt_path: Optional[Path] = None):
        try:
            from cryptography.fernet import Fernet, MultiFernet
        except ImportError as exc:  # pragma: no cover
            raise EncryptionConfigurationError(
                "MEMORY_ENCRYPTION_ENABLED requires the optional `cryptography` dependency: "
                'pip install "llm-memory[crypto]"'
            ) from exc

        keys = self._build_fernet_keys(key, salt_path)
        self._fernet = MultiFernet([Fernet(item) for item in keys])

    @classmethod
    def _derive(cls, key_bytes: bytes, salt: bytes) -> bytes:
        derived = hashlib.pbkdf2_hmac("sha256", key_bytes, salt, cls._KDF_ITERATIONS, dklen=32)
        return base64.urlsafe_b64encode(derived)

    @classmethod
    def _build_fernet_keys(cls, key: str, salt_path: Optional[Path] = None) -> list[bytes]:
        """Return the encryption key first, followed by keys accepted only for decryption."""

        key_bytes = key.encode("utf-8")

        if len(key_bytes) == 44:
            try:
                decoded = base64.urlsafe_b64decode(key_bytes)
            except (binascii.Error, ValueError):
                decoded = b""
            if len(decoded) == 32:
                return [key_bytes]

        if len(key_bytes) < cls._PASSPHRASE_MIN_LENGTH:
            raise EncryptionConfigurationError(
                "Encryption key must be a valid Fernet key or a passphrase with at least 32 bytes."
            )

        legacy_key = cls._derive(key_bytes, cls._LEGACY_KDF_SALT)
        if salt_path is None:
            return [legacy_key]
        # The legacy static-salt key stays in the list so payloads written by
        # earlier versions remain readable; new payloads use the random salt.
        return [cls._derive(key_bytes, load_or_create_salt(salt_path)), legacy_key]

    def encrypt(self, plaintext: str) -> CipherResult:
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return CipherResult(payload=token.decode("utf-8"), encrypted=True)

    def decrypt(self, payload: str) -> str:
        from cryptography.fernet import InvalidToken

        try:
            return self._fernet.decrypt(payload.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise PayloadDecryptionError("Invalid encrypted payload.") from exc


def build_cipher(enabled: bool, key_env: str, salt_path: Optional[Path] = None) -> PayloadCipher:
    """Build the optional cipher using a key read from the environment."""

    if not enabled:
        return NoOpCipher()

    key = os.getenv(key_env)
    if not key:
        raise EncryptionConfigurationError(
            f"Encryption is enabled but the environment variable `{key_env}` is not set."
        )

    return FernetCipher(key, salt_path=salt_path)


def default_salt_path(sqlite_db_path: Path) -> Path:
    db_path = Path(sqlite_db_path)
    return db_path.with_name(f"{db_path.name}.salt")
