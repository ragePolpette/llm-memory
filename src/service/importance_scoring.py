"""Deterministic self-evaluation scoring for memory.add (experimental)."""

from __future__ import annotations

import hashlib
import json
from math import log1p
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
    novelty_status: str = "computed",
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

    if novelty_status == "failed" or not novelty_computed:
        novelty_score: Optional[float] = None
        novelty_signal = 0.0
    elif not top_similarities:
        novelty_score = 1.0
        novelty_signal = novelty_score
    else:
        novelty_score = 1.0 - _clamp01(max(top_similarities), default=0.0)
        novelty_score = _clamp01(novelty_score, default=0.0)
        novelty_signal = novelty_score

    if surprise_source == "confidence":
        base = 0.45 * surprise_score + 0.35 * novelty_signal + 0.20 * inference_score
    else:
        base = 0.20 * surprise_score + 0.45 * novelty_signal + 0.35 * inference_score

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
            "novelty_score": round(novelty_score, 6) if novelty_score is not None else None,
            "novelty_status": novelty_status,
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


def _fast_static_score(metadata: dict[str, Any]) -> float:
    importance_score = metadata.get("importance_score")
    importance_signal = None
    if importance_score is not None:
        try:
            importance_signal = _clamp01(float(importance_score) / 100.0, default=0.0)
        except (TypeError, ValueError):
            importance_signal = None

    novelty_signal = None
    if metadata.get("novelty_score") is not None:
        novelty_signal = _clamp01(metadata.get("novelty_score"), default=0.0)

    confidence_source = metadata.get("confidence")
    if confidence_source is None:
        confidence_source = metadata.get("predictive_confidence")
    if confidence_source is None:
        confidence_source = metadata.get("predictive_confidence_before")
    confidence_signal = None
    if confidence_source is not None:
        confidence_signal = _clamp01(confidence_source, default=0.0)

    impact_signal = None
    if metadata.get("negative_impact") is not None:
        impact_signal = _clamp01(metadata.get("negative_impact"), default=0.0)

    weighted_total = 0.0
    weights = 0.0
    for signal, weight in (
        (importance_signal, 0.45),
        (novelty_signal, 0.25),
        (confidence_signal, 0.20),
        (impact_signal, 0.10),
    ):
        if signal is None:
            continue
        weighted_total += signal * weight
        weights += weight

    if weights == 0.0:
        return 0.35
    return _clamp01(weighted_total / weights, default=0.35)


def _fast_frequency_score(recurrence_count: int) -> float:
    occurrences = max(1, int(recurrence_count))
    if occurrences <= 1:
        return 0.0
    return _clamp01(log1p(occurrences - 1) / log1p(7), default=0.0)


def _fast_quality_score(metadata: dict[str, Any], recurrence_count: int) -> float:
    signals: list[float] = []

    for key, ceiling in (
        ("distinct_session_count", 4),
        ("distinct_task_count", 4),
        ("distinct_day_count", 3),
        ("outcome_reuse_count", 3),
        ("distinct_entity_count", 5),
    ):
        value = metadata.get(key)
        if value is None:
            continue
        try:
            numeric = max(0, int(value))
        except (TypeError, ValueError):
            continue
        signals.append(_clamp01(numeric / ceiling, default=0.0))

    for key in ("time_spread_score", "entity_spread_score", "semantic_cohesion", "scope_alignment_score"):
        value = metadata.get(key)
        if value is None:
            continue
        signals.append(_clamp01(value, default=0.0))

    base_quality = 0.35 if int(recurrence_count) > 1 else 0.0
    if not signals:
        return base_quality
    return _clamp01(max(base_quality, sum(signals) / len(signals)), default=base_quality)


def _fast_noise_penalty(metadata: dict[str, Any], *, event_type: str) -> float:
    direct_penalty = metadata.get("noise_penalty")
    if direct_penalty is not None:
        return _clamp01(direct_penalty, default=0.0)

    penalties: list[float] = []
    if str(event_type).strip().lower() == "retry":
        penalties.append(0.15)

    for key in ("duplicate_ratio", "same_session_ratio", "loop_ratio"):
        value = metadata.get(key)
        if value is None:
            continue
        penalties.append(_clamp01(value, default=0.0))

    semantic_cohesion = metadata.get("semantic_cohesion")
    if semantic_cohesion is not None:
        penalties.append(_clamp01(1.0 - float(semantic_cohesion), default=0.0))

    scope_alignment = metadata.get("scope_alignment_score")
    if scope_alignment is not None:
        penalties.append(_clamp01(1.0 - float(scope_alignment), default=0.0))

    burst_retry_count = metadata.get("burst_retry_count")
    if burst_retry_count is not None:
        penalties.append(_clamp01(int(burst_retry_count) / 5.0, default=0.0))

    if not penalties:
        return 0.0
    return _clamp01(sum(penalties) / len(penalties), default=0.0)


def build_fast_selection_metadata(
    *,
    metadata: dict[str, Any] | None,
    recurrence_count: int,
    event_type: str = "note",
    alpha: float = 0.65,
) -> dict[str, Any]:
    normalized_metadata = dict(metadata or {})
    meta_score = _fast_static_score(normalized_metadata)
    frequency_score = _fast_frequency_score(recurrence_count)
    quality_score = _fast_quality_score(normalized_metadata, recurrence_count)
    recurrence_score = _clamp01(frequency_score * quality_score, default=0.0)
    noise_penalty = _fast_noise_penalty(normalized_metadata, event_type=event_type)
    recurrence_boost = _clamp01(
        _clamp01(alpha, default=0.65)
        * recurrence_score
        * (1.0 - meta_score)
        * (1.0 - noise_penalty),
        default=0.0,
    )
    selection_score = _clamp01(meta_score + recurrence_boost, default=0.0)

    return {
        "formula_version": "fast-memory-v1",
        "alpha": round(_clamp01(alpha, default=0.65), 6),
        "meta_score": round(meta_score, 6),
        "recurrence_frequency": round(frequency_score, 6),
        "recurrence_quality": round(quality_score, 6),
        "recurrence_score": round(recurrence_score, 6),
        "noise_penalty": round(noise_penalty, 6),
        "recurrence_boost": round(recurrence_boost, 6),
        "selection_score": round(selection_score, 6),
    }

