"""Secret detection for writes.

A memory is read back by agents and sent to the embedding provider, so a secret in it
leaks twice. Writes that contain one are rejected, not redacted: the caller learns
which field and which kind of secret, never the value, and writes again without it.

Patterns favour precision. A missed secret is possible (no detector catches all of
them); a false rejection costs an agent one retry with the value removed, which it can
always do because the error names the field and the kind.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

# (kind, pattern). Each pattern matches the secret value itself.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA|ABIA|ACCA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b")),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("slack_webhook", re.compile(r"https://hooks\.slack\.com/services/[A-Za-z0-9/]{20,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("fireworks_key", re.compile(r"\bfw_[A-Za-z0-9]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("npm_token", re.compile(r"\bnpm_[A-Za-z0-9]{36}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{16,}")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/-]{24,}=*)")),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.-]{1,20}://[^\s/:@]{1,64}:([^\s/@]{3,128})@[^\s/]+")),
]

# key = value, key: value, "key": "value" for secret-sounding keys; the value is checked
# for placeholders and entropy, since prose like "password: see the vault" is common.
_ASSIGNMENT = re.compile(
    r"(?i)\b([a-z0-9_.-]*(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|private[_-]?key"
    r"|client[_-]?secret|auth[_-]?key)[a-z0-9_.-]*)[\"']?\s*(?:=|:|=>)\s*[\"']?([^\s\"',;}{)(]{8,})"
)

_PLACEHOLDER = re.compile(
    r"(?i)^(?:<[^>]*>|\$\{[^}]*\}|\$[A-Z_][A-Z0-9_]*|%[^%]+%|\{\{[^}]*\}\}|x{3,}|\*{3,}|\.{3,}|changeme|redacted"
    r"|\[redacted[^\]]*\]|your[_-].*|example.*|dummy.*|placeholder.*|none|null|true|false|undefined|os\.environ.*"
    r"|env\(.*|process\.env.*|getenv.*|settings\..*|config\..*)$"
)


@dataclass(frozen=True)
class SecretFinding:
    field: str
    kind: str


def _entropy(value: str) -> float:
    counts = {ch: value.count(ch) for ch in set(value)}
    return -sum(c / len(value) * math.log2(c / len(value)) for c in counts.values())


def _looks_secret(value: str) -> bool:
    if _PLACEHOLDER.match(value) or len(set(value)) < 5:
        return False
    has_digit = any(ch.isdigit() for ch in value)
    has_alpha = any(ch.isalpha() for ch in value)
    return has_digit and has_alpha and _entropy(value) >= 3.0


def scan_text(text: str) -> list[str]:
    """Kinds of secrets found in one text (empty when clean)."""

    if not text:
        return []
    kinds: list[str] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(match.lastindex or 0)
            if kind in ("bearer_token", "url_credentials") and _PLACEHOLDER.match(value):
                continue
            kinds.append(kind)
            break
    for match in _ASSIGNMENT.finditer(text):
        if _looks_secret(match.group(2)):
            kinds.append("credential_assignment")
            break
    return list(dict.fromkeys(kinds))


def scan_fields(fields: Iterable[tuple[str, object]]) -> list[SecretFinding]:
    """Scan named fields; lists and dicts are scanned value by value (dict keys included)."""

    findings: list[SecretFinding] = []

    def visit(name: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, str):
            findings.extend(SecretFinding(name, kind) for kind in scan_text(value))
        elif isinstance(value, dict):
            for key, item in value.items():
                # "api_key": "abc123..." as a dict entry is the assignment pattern too
                if isinstance(item, str):
                    findings.extend(SecretFinding(f"{name}.{key}", kind) for kind in scan_text(f"{key}={item}"))
                else:
                    visit(f"{name}.{key}", item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(f"{name}[{index}]", item)

    for name, value in fields:
        visit(name, value)
    return list(dict.fromkeys(findings))
