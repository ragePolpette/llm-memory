"""Deterministic self-evaluation scoring for memory.add (experimental)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from ..config import MemoryScope
from ..models import ScopeRef


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _clamp_int(value: Any, minimum: int, maximum: int, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def _norm_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _stable_fingerprint(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    canonical = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def _normalized_retrieved_ids(raw_ids: Any) -> list[str]:
    if not isinstance(raw_ids, list):
        return []
    values = [_norm_str(item) for item in raw_ids]
    values = [item for item in values if item]
    return sorted(set(values))


def _resolve_writer_model(
    payload: dict[str, Any],
    *,
    actor_agent_id: str,
    runtime_writer_model: Optional[str],
) -> str:
    runtime_model = _norm_str(runtime_writer_model)
    if runtime_model:
        return runtime_model

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    candidate = _norm_str(payload.get("writer_model")) or _norm_str(metadata.get("writer_model"))
    if candidate:
        return candidate

    normalized_agent = _norm_str(actor_agent_id).lower()
    if "sonnet" in normalized_agent or "claude" in normalized_agent:
        return "sonnet"
    if "gpt" in normalized_agent or "openai" in normalized_agent:
        return "gpt"
    if "gemini" in normalized_agent:
        return "gemini"
    return "unknown-model"


def _resolve_scope_label(payload: dict[str, Any], visibility: MemoryScope) -> str:
    scope_label = _norm_str(payload.get("scope_label")).lower()
    if scope_label in {"shared", "project", "agent"}:
        return scope_label
    if visibility == MemoryScope.PRIVATE:
        return "agent"
    if visibility == MemoryScope.SHARED:
        return "shared"
    return "project"


def _context_fields(payload: dict[str, Any]) -> dict[str, Any]:
    context_fingerprint = payload.get("context_fingerprint")
    if not isinstance(context_fingerprint, dict):
        context_fingerprint = {}

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    conversation_id = (
        _norm_str(context_fingerprint.get("conversation_id"))
        or _norm_str(metadata.get("conversation_id"))
        or _norm_str(metadata.get("session_id"))
        or _norm_str(payload.get("session_id"))
    )
    task_id = _norm_str(context_fingerprint.get("task_id")) or _norm_str(metadata.get("task_id"))
    retrieved_ids = _normalized_retrieved_ids(context_fingerprint.get("retrieved_ids"))
    tool_trace_fingerprint = _stable_fingerprint(context_fingerprint.get("tool_trace_fingerprint"))
    prompt_fingerprint = _stable_fingerprint(context_fingerprint.get("prompt_fingerprint"))

    return {
        "conversation_id": conversation_id,
        "task_id": task_id,
        "retrieved_ids": retrieved_ids,
        "tool_trace_fingerprint": tool_trace_fingerprint,
        "prompt_fingerprint": prompt_fingerprint,
    }


def _compute_context_hash(
    *,
    project_id: str,
    scope_label: str,
    writer_model: str,
    context_fields: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    canonical_context = {
        "project_id": project_id,
        "scope": scope_label,
        "conversation_id": context_fields["conversation_id"],
        "task_id": context_fields["task_id"],
        "retrieved_ids": context_fields["retrieved_ids"],
        "tool_trace_fingerprint": context_fields["tool_trace_fingerprint"],
        "prompt_fingerprint": context_fields["prompt_fingerprint"],
        "writer_model": writer_model,
    }
    serialized = json.dumps(canonical_context, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return digest, canonical_context


def _importance_payload(payload: dict[str, Any]) -> dict[str, Any]:
    importance = payload.get("importance")
    if isinstance(importance, dict):
        return importance
    return {}


def has_surprise_signal(payload: dict[str, Any]) -> bool:
    importance = _importance_payload(payload)
    return any(
        importance.get(key) is not None
        for key in (
            "confidence",
            "predictive_confidence",
            "predictive_confidence_before",
            "proxy_disagreement",
            "disagreement_score",
            "self_rating",
            "surprise_self_rating",
        )
    )


def has_inference_signal(payload: dict[str, Any]) -> bool:
    importance = _importance_payload(payload)
    return any(
        value is not None
        for value in (
            payload.get("tool_steps"),
            payload.get("correction_count"),
            payload.get("inference_level"),
            importance.get("tool_steps"),
            importance.get("correction_count"),
            importance.get("inference_level"),
            importance.get("inference_steps"),
        )
    )


def _compute_surprise(payload: dict[str, Any]) -> tuple[float, str, str]:
    importance = _importance_payload(payload)
    confidence = importance.get("confidence")
    if confidence is None:
        confidence = importance.get("predictive_confidence")
    if confidence is None:
        confidence = importance.get("predictive_confidence_before")
    if confidence is not None:
        score = 1.0 - _clamp01(confidence, default=0.5)
        return score, "confidence", "high"

    disagreement = importance.get("proxy_disagreement")
    if disagreement is None:
        disagreement = importance.get("disagreement_score")
    if disagreement is not None:
        return _clamp01(disagreement, default=0.5), "disagreement", "medium"

    self_rating = importance.get("self_rating")
    if self_rating is None:
        self_rating = importance.get("surprise_self_rating")
    if self_rating is not None:
        return _clamp01(self_rating, default=0.5), "self", "low"

    return 0.5, "self", "low"


def _compute_inference(payload: dict[str, Any]) -> tuple[float, int, int, int]:
    importance = _importance_payload(payload)
    tool_steps = importance.get("tool_steps")
    if tool_steps is None:
        tool_steps = payload.get("tool_steps")

    correction_count = importance.get("correction_count")
    if correction_count is None:
        correction_count = payload.get("correction_count")

    inference_level = importance.get("inference_level")
    if inference_level is None:
        inference_level = payload.get("inference_level")
    if inference_level is None:
        inference_level = importance.get("inference_steps")

    tool_steps_clamped = _clamp_int(tool_steps, minimum=0, maximum=10, default=0)
    corrections_clamped = _clamp_int(correction_count, minimum=0, maximum=5, default=0)
    inference_level_clamped = _clamp_int(inference_level, minimum=0, maximum=5, default=0)

    normalized = (tool_steps_clamped + corrections_clamped + inference_level_clamped) / 20.0
    return _clamp01(normalized, default=0.0), inference_level_clamped, tool_steps_clamped, corrections_clamped


def _compute_negative_impact(payload: dict[str, Any]) -> float:
    importance = _importance_payload(payload)
    value = importance.get("negative_impact")
    if value is None:
        value = payload.get("negative_impact")
    return _clamp01(value, default=0.0)


def build_importance_metadata(
    *,
    payload: dict[str, Any],
    scope: ScopeRef,
    visibility: MemoryScope,
    top_similarities: list[float],
    novelty_computed: bool,
    event_ts_utc: str,
    actor_agent_id: str,
    runtime_writer_model: Optional[str] = None,
) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)

    writer_model = _resolve_writer_model(
        payload=payload,
        actor_agent_id=actor_agent_id,
        runtime_writer_model=runtime_writer_model,
    )
    scope_label = _resolve_scope_label(payload, visibility)
    context_fields = _context_fields(payload)
    context_hash, canonical_context = _compute_context_hash(
        project_id=scope.project_id,
        scope_label=scope_label,
        writer_model=writer_model,
        context_fields=context_fields,
    )

    surprise_score, surprise_source, signal_quality = _compute_surprise(payload)
    inference_score, inference_level, tool_steps, correction_count = _compute_inference(payload)

    if not novelty_computed:
        novelty_score = 1.0
    elif not top_similarities:
        novelty_score = 1.0
    else:
        novelty_score = 1.0 - _clamp01(max(top_similarities), default=0.0)
    novelty_score = _clamp01(novelty_score, default=1.0)

    if surprise_source == "confidence":
        base = 0.45 * surprise_score + 0.35 * novelty_score + 0.20 * inference_score
    else:
        base = 0.20 * surprise_score + 0.45 * novelty_score + 0.35 * inference_score

    negative_impact = _compute_negative_impact(payload)
    score_with_neg = base + 0.25 * negative_impact
    importance_score = int(max(0, min(100, round(score_with_neg * 100))))
    if importance_score >= 70:
        importance_class = "high"
    elif importance_score >= 40:
        importance_class = "medium"
    else:
        importance_class = "low"

    importance = _importance_payload(payload)
    external_raw = importance.get("is_external")
    if external_raw is None:
        external_raw = payload.get("is_external")

    metadata.update(
        {
            "event_ts_utc": event_ts_utc,
            "context_hash": context_hash,
            "context_fingerprint": canonical_context,
            "writer_model": writer_model,
            "writer_agent_id": actor_agent_id,
            "scope": scope_label,
            "surprise_source": surprise_source,
            "signal_quality": signal_quality,
            "surprise_score": round(surprise_score, 6),
            "novelty_score": round(novelty_score, 6),
            "inference_score": round(inference_score, 6),
            "inference_level": inference_level,
            "tool_steps": tool_steps,
            "correction_count": correction_count,
            "negative_impact": round(negative_impact, 6),
            "importance_score": importance_score,
            "importance_class": importance_class,
            "is_external": bool(external_raw),
            "novelty_computed": bool(novelty_computed),
            "self_eval_policy_mode": "experimental-v1",
        }
    )
    return metadata

