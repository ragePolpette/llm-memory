from __future__ import annotations

from base64 import urlsafe_b64encode
from os import urandom

import pytest

from src.security.crypto import EncryptionConfigurationError, FernetCipher


def test_fernet_cipher_rejects_short_passphrases():
    with pytest.raises(EncryptionConfigurationError, match="at least 32 bytes"):
        FernetCipher("too-short-secret")


def test_fernet_cipher_derives_key_from_long_passphrase():
    pytest.importorskip("cryptography.fernet")

    cipher = FernetCipher("x" * 32)
    encrypted = cipher.encrypt("payload")

    assert encrypted.encrypted is True
    assert cipher.decrypt(encrypted.payload) == "payload"


def test_fernet_cipher_accepts_valid_fernet_key_verbatim():
    pytest.importorskip("cryptography.fernet")

    raw_key = urlsafe_b64encode(urandom(32)).decode("utf-8")
    cipher = FernetCipher(raw_key)
    encrypted = cipher.encrypt("payload")

    assert cipher.decrypt(encrypted.payload) == "payload"
