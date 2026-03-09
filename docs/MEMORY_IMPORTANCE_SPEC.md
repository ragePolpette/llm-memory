# Memory Self-Eval Scoring Spec (experimental-v1)

## Scope
- Runtime path: `memory.add`
- Status: experimental, non-hardened
- Enforcement toggle: `MEMORY_SELF_EVAL_ENFORCED`

## Scoring
- Surprise:
  - `confidence` -> `1 - confidence` (`high`)
  - `proxy_disagreement` -> value (`medium`)
  - `self_rating` -> value (`low`)
- Novelty:
  - `1 - max_similarity(top_k=5)`
  - if no memories: `1.0`
  - if similarity cannot be computed: `1.0`, `novelty_computed=false`
- Inference:
  - normalize `tool_steps + correction_count + inference_level`
  - max: `10 + 5 + 5`
- Base weights:
  - confidence: `0.45 surprise + 0.35 novelty + 0.20 inference`
  - disagreement/self: `0.20 surprise + 0.45 novelty + 0.35 inference`
- Negative impact:
  - `score_with_neg = base + 0.25 * negative_impact`
- Final score:
  - `importance_score = clip(round(score_with_neg * 100), 0, 100)`
  - classes: `low (0-39)`, `medium (40-69)`, `high (70-100)`

## Mandatory metadata (persisted)
- `event_ts_utc`
- `context_hash` (sha256 canonical truncated to 16 chars)
- `writer_model`
- `writer_agent_id`
- `scope` (`shared|project|agent`)
- `surprise_source`
- `signal_quality`
- `surprise_score`
- `novelty_score`
- `inference_score`
- `inference_level`
- `negative_impact`
- `importance_score`
- `importance_class`
- `is_external`
- `novelty_computed`

## Context hash canonical fields
- `project_id`
- `scope`
- `conversation_id`
- `task_id`
- `retrieved_ids` (sorted + deduped)
- `tool_trace_fingerprint`
- `prompt_fingerprint`
- `writer_model`

## Bias-loop mitigation
- `novelty_score < 0.2` should be excluded from FT dataset.
- Keep external quota (`is_external=true`) in dataset selection.
- Apply bucket mix for fine-tuning selection (top/mid/low).
