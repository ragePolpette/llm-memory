"""Policy privacy/redaction locale."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrivacyDecision:
    """Decisione applicata al contenuto prima della persistenza."""

    content: str
    metadata: dict
    redacted: bool
    should_encrypt: bool


class PrivacyPolicy:
    """Applica policy su campi sensibili e metadati."""

    def __init__(
        self,
        sensitive_tags: list[str],
        drop_metadata_keys: list[str],
        encrypt_sensitive: bool,
    ):
        self.sensitive_tags = {tag.lower() for tag in sensitive_tags}
        self.drop_metadata_keys = {key.lower() for key in drop_metadata_keys}
        self.encrypt_sensitive = encrypt_sensitive

    def apply(self, content: str, metadata: dict, sensitivity_tags: list[str]) -> PrivacyDecision:
        tags = {tag.lower() for tag in sensitivity_tags}
        intersects_sensitive = bool(tags.intersection(self.sensitive_tags))

        sanitized_metadata = {
            key: value
            for key, value in metadata.items()
            if key.lower() not in self.drop_metadata_keys
        }

        if not intersects_sensitive:
            return PrivacyDecision(
                content=content,
                metadata=sanitized_metadata,
                redacted=False,
                should_encrypt=False,
            )

        if self.encrypt_sensitive:
            return PrivacyDecision(
                content=content,
                metadata=sanitized_metadata,
                redacted=False,
                should_encrypt=True,
            )

        return PrivacyDecision(
            content="[REDACTED:SENSITIVE]",
            metadata=sanitized_metadata,
            redacted=True,
            should_encrypt=False,
        )
