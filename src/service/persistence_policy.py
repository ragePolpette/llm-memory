"""Centralized deny-by-default persistence policy for memory writes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .memory_service import ActorContext


POLICY_VERSION = "persistence-v1"
INTERNAL_POLICY_VERSION = "internal-persistence-v1"
_WS_RE = re.compile(r"\s+")
_CODE_LINE_RE = re.compile(
    r"^\s*(def |class |return |if |for |while |try:|except |SELECT |INSERT |UPDATE |DELETE |FROM |WHERE |public |private |function |\{|\})",
    re.IGNORECASE,
)

_SMALL_TALK_EXACT = {
    "ciao",
    "come va",
    "grazie",
    "hello",
    "hey",
    "hi",
    "perfetto",
    "salve",
}
_NOISE_EXACT = {"ok", "test", "va bene", "ricevuto"}
_TRANSIENT_HINTS = (
    "questa sessione",
    "questa chat",
    "in questa chat",
    "sessione corrente",
    "contesto temporaneo",
    "temporary context",
    "solo per ora",
    "ricordamelo dopo",
)
_TASK_PROGRESS_HINTS = (
    "ho finito",
    "sto lavorando",
    "in lavorazione",
    "wip",
    "task completato",
    "task completion",
    "done",
    "completed",
    "next step",
    "todo",
)
_DEBUG_HINTS = (
    "debug",
    "debugging",
    "stack trace",
    "traceback",
    "exception",
    "temporary note",
    "nota di debug",
    "raw note",
)
_WORKING_MEMORY_HINTS = (
    "devo controllare",
    "devo verificare",
    "da fare",
    "to do",
    "check later",
    "provare",
    "ipotesi temporanea",
    "working note",
)
_PREFERENCE_HINTS = (
    "preferisco",
    "preferiamo",
    "usa sempre",
    "usare sempre",
    "evita sempre",
    "non voglio",
    "non usare",
)
_ARCHITECTURE_HINTS = (
    "decisione architetturale",
    "architettura",
    "architetturale",
    "pattern",
    "standard",
    "scegliamo",
    "usiamo",
    "adottiamo",
)
_SEMANTIC_HINTS = (
    "il sistema",
    "il progetto",
    "usa ",
    "supporta",
    "richiede",
    "vincolo",
    "regola",
    "deve ",
    "non deve",
    "predefinito",
    "default",
)


@dataclass(frozen=True)
class PersistenceDecision:
    accepted: bool
    category: str | None
    reason_codes: list[str] = field(default_factory=list)
    confidence: float | None = None
    policy_version: str = POLICY_VERSION
    normalized_summary: str | None = None
    source_type: str = "user"
    internal_reason: str | None = None

    @property
    def decision(self) -> str:
        if self.accepted and self.source_type == "internal_governance":
            return "accepted_internal"
        return "accepted" if self.accepted else "rejected"

    def as_payload(self, *, write_path: str) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "accepted": self.accepted,
            "category": self.category,
            "reason_codes": list(self.reason_codes),
            "confidence": self.confidence,
            "policy_version": self.policy_version,
            "normalized_summary": self.normalized_summary,
            "source": self.source_type,
            "source_type": self.source_type,
            "internal_reason": self.internal_reason,
            "write_path": write_path,
        }


def _normalize_text(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "").strip())


def _normalized_summary(content: str) -> str:
    if len(content) <= 180:
        return content
    return content[:177].rstrip() + "..."


def _has_any(text: str, hints: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in hints)


def _looks_like_code(text: str) -> bool:
    if "```" in text:
        return True

    syntax_hits = sum(
        marker in text
        for marker in (
            "();",
            "=>",
            "::",
            "{",
            "}",
            "SELECT ",
            "INSERT ",
            "UPDATE ",
            "DELETE ",
            "<div",
            "</",
        )
    )
    code_lines = sum(1 for line in text.splitlines() if _CODE_LINE_RE.match(line))
    return syntax_hits >= 3 or code_lines >= 2


def classify_persistence(payload: dict[str, Any], actor: "ActorContext", *, write_path: str) -> PersistenceDecision:
    raw_content = payload.get("content")
    content = _normalize_text(raw_content)
    context = _normalize_text(payload.get("context"))
    tags = " ".join(str(tag) for tag in payload.get("tags", []) if tag)
    composite = " ".join(part for part in (content, context, tags) if part).strip()
    composite_lower = composite.lower()
    requested_type = _normalize_text(payload.get("type")).lower()
    summary = _normalized_summary(content or composite)

    if not content:
        return PersistenceDecision(
            accepted=False,
            category="noise",
            reason_codes=["EMPTY_CONTENT", "DEFAULT_DENY"],
            confidence=1.0,
            normalized_summary=summary,
        )

    if content.lower() in _SMALL_TALK_EXACT or content.lower().startswith(("ciao ", "come va ", "hello ")):
        return PersistenceDecision(
            accepted=False,
            category="small_talk",
            reason_codes=["SMALL_TALK", "NOT_REUSABLE", "DEFAULT_DENY"],
            confidence=0.99,
            normalized_summary=summary,
        )

    if content.lower() in _NOISE_EXACT or len(content) < 4:
        return PersistenceDecision(
            accepted=False,
            category="noise",
            reason_codes=["LOW_SIGNAL", "NOT_REUSABLE", "DEFAULT_DENY"],
            confidence=0.97,
            normalized_summary=summary,
        )

    if _looks_like_code(raw_content or content):
        return PersistenceDecision(
            accepted=False,
            category="code_snippet",
            reason_codes=["CODE_SNIPPET", "NOT_REUSABLE", "DEFAULT_DENY"],
            confidence=0.98,
            normalized_summary=summary,
        )

    if _has_any(composite_lower, _TASK_PROGRESS_HINTS):
        return PersistenceDecision(
            accepted=False,
            category="task_progress",
            reason_codes=["TASK_PROGRESS", "SESSION_LOCAL", "DEFAULT_DENY"],
            confidence=0.95,
            normalized_summary=summary,
        )

    if _has_any(composite_lower, _TRANSIENT_HINTS):
        return PersistenceDecision(
            accepted=False,
            category="transient_context",
            reason_codes=["TRANSIENT_CONTEXT", "SESSION_LOCAL", "DEFAULT_DENY"],
            confidence=0.95,
            normalized_summary=summary,
        )

    if _has_any(composite_lower, _DEBUG_HINTS):
        return PersistenceDecision(
            accepted=False,
            category="working_memory",
            reason_codes=["DEBUG_NOTE", "NOT_REUSABLE", "DEFAULT_DENY"],
            confidence=0.94,
            normalized_summary=summary,
        )

    if _has_any(composite_lower, _WORKING_MEMORY_HINTS):
        return PersistenceDecision(
            accepted=False,
            category="working_memory",
            reason_codes=["WORKING_MEMORY", "NOT_REUSABLE", "DEFAULT_DENY"],
            confidence=0.9,
            normalized_summary=summary,
        )

    if _has_any(composite_lower, _PREFERENCE_HINTS):
        return PersistenceDecision(
            accepted=True,
            category="stable_preference",
            reason_codes=["STABLE_PREFERENCE", "BEYOND_SESSION"],
            confidence=0.88,
            normalized_summary=summary,
        )

    if requested_type == "decision" or _has_any(composite_lower, _ARCHITECTURE_HINTS):
        return PersistenceDecision(
            accepted=True,
            category="architectural_decision",
            reason_codes=["ARCHITECTURAL_DECISION", "REUSABLE_RULE"],
            confidence=0.84,
            normalized_summary=summary,
        )

    if len(content) >= 25 and _has_any(composite_lower, _SEMANTIC_HINTS):
        return PersistenceDecision(
            accepted=True,
            category="semantic_memory",
            reason_codes=["REUSABLE_FACT", "BEYOND_SESSION"],
            confidence=0.78,
            normalized_summary=summary,
        )

    return PersistenceDecision(
        accepted=False,
        category="working_memory",
        reason_codes=["DEFAULT_DENY", "INSUFFICIENT_PERSISTENCE_SIGNAL"],
        confidence=0.7,
        normalized_summary=summary,
    )


def classify_internal_write(
    *,
    record_type: str,
    content: str,
    internal_reason: str,
    write_path: str,
) -> PersistenceDecision:
    normalized_reason = _normalize_text(internal_reason)
    normalized_content = _normalize_text(content)
    normalized_record_type = _normalize_text(record_type).lower()
    summary = _normalized_summary(normalized_content)

    if not normalized_reason:
        return PersistenceDecision(
            accepted=False,
            category="internal_governance",
            reason_codes=["MISSING_INTERNAL_REASON"],
            confidence=1.0,
            policy_version=INTERNAL_POLICY_VERSION,
            normalized_summary=summary,
            source_type="internal_governance",
        )

    if not normalized_content:
        return PersistenceDecision(
            accepted=False,
            category="internal_governance",
            reason_codes=["EMPTY_INTERNAL_CONTENT"],
            confidence=1.0,
            policy_version=INTERNAL_POLICY_VERSION,
            normalized_summary=summary,
            source_type="internal_governance",
            internal_reason=normalized_reason,
        )

    reason_code = normalized_record_type.upper() if normalized_record_type else "INTERNAL_RECORD"
    return PersistenceDecision(
        accepted=True,
        category=f"internal_{normalized_record_type or 'record'}",
        reason_codes=["INTERNAL_GOVERNANCE", reason_code],
        confidence=1.0,
        policy_version=INTERNAL_POLICY_VERSION,
        normalized_summary=summary,
        source_type="internal_governance",
        internal_reason=normalized_reason,
    )
