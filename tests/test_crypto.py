from __future__ import annotations

from base64 import urlsafe_b64encode
from os import urandom

import pytest

from src.security.crypto import EncryptionConfigurationError, FernetCipher, PayloadDecryptionError


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


def test_fernet_cipher_raises_specific_error_for_invalid_payload():
    pytest.importorskip("cryptography.fernet")

    cipher = FernetCipher("x" * 32)

    with pytest.raises(PayloadDecryptionError, match="Invalid encrypted payload"):
        cipher.decrypt("not-a-valid-token")


def test_passphrase_key_uses_random_persisted_salt(tmp_path):
    pytest.importorskip("cryptography.fernet")
    from src.security.crypto import load_or_create_salt

    salt_a = tmp_path / "a" / "memory.db.salt"
    salt_b = tmp_path / "b" / "memory.db.salt"
    cipher_a = FernetCipher("x" * 32, salt_path=salt_a)
    cipher_b = FernetCipher("x" * 32, salt_path=salt_b)

    assert salt_a.exists() and salt_b.exists()
    assert load_or_create_salt(salt_a) != load_or_create_salt(salt_b)
    assert (salt_a.stat().st_mode & 0o777) == 0o600

    token = cipher_a.encrypt("payload").payload
    with pytest.raises(PayloadDecryptionError):
        cipher_b.decrypt(token)
    assert FernetCipher("x" * 32, salt_path=salt_a).decrypt(token) == "payload"


def test_salted_cipher_still_reads_legacy_static_salt_payloads(tmp_path):
    pytest.importorskip("cryptography.fernet")

    legacy_token = FernetCipher("x" * 32).encrypt("legacy").payload
    salted = FernetCipher("x" * 32, salt_path=tmp_path / "memory.db.salt")

    assert salted.decrypt(legacy_token) == "legacy"
    new_token = salted.encrypt("fresh").payload
    with pytest.raises(PayloadDecryptionError):
        FernetCipher("x" * 32).decrypt(new_token)


def test_corrupted_salt_file_is_rejected(tmp_path):
    from src.security.crypto import load_or_create_salt

    salt_path = tmp_path / "memory.db.salt"
    salt_path.write_text("not-hex", encoding="ascii")
    with pytest.raises(EncryptionConfigurationError, match="corrupted"):
        load_or_create_salt(salt_path)
    salt_path.write_text("abcd", encoding="ascii")
    with pytest.raises(EncryptionConfigurationError, match="too short"):
        load_or_create_salt(salt_path)


def test_build_cipher_places_salt_next_to_database(tmp_path, monkeypatch):
    pytest.importorskip("cryptography.fernet")
    from src.security.crypto import build_cipher, default_salt_path

    monkeypatch.setenv("TEST_MEMORY_KEY", "y" * 40)
    salt_path = default_salt_path(tmp_path / "memory.db")
    cipher = build_cipher(True, "TEST_MEMORY_KEY", salt_path=salt_path)

    assert salt_path == tmp_path / "memory.db.salt"
    assert salt_path.exists()
    assert cipher.decrypt(cipher.encrypt("v").payload) == "v"

    monkeypatch.delenv("TEST_MEMORY_KEY")
    with pytest.raises(EncryptionConfigurationError, match="not set"):
        build_cipher(True, "TEST_MEMORY_KEY", salt_path=salt_path)


def test_pyproject_declares_cryptography_as_optional_extra():
    import tomllib
    from pathlib import Path

    data = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    optional = data["project"]["optional-dependencies"]
    assert any(dep.startswith("cryptography") for dep in optional["crypto"])
    assert not any(dep.startswith("cryptography") for dep in data["project"]["dependencies"])
