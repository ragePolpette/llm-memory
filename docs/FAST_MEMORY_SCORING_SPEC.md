# Fast Memory Selection Scoring Spec

## Scope

This spec applies only to fast memory.

It does not modify:

- `memory.add`
- strong-memory ranking
- semantic retrieval ranking for durable entries

## Formula V1

```text
S_select(m) = S_meta(m) + alpha * R(m) * (1 - S_meta(m)) * (1 - N(m))
```

## Variables

- `S_meta(m)` = normalized static score in `[0, 1]`
- `R(m)` = normalized recurrence score in `[0, 1]`
- `N(m)` = normalized noise penalty in `[0, 1]`
- `alpha` = recurrence trust coefficient

## Recurrence

```text
R(m) = F(m) * Q(m)
```

- `F(m)` = frequency strength
- `Q(m)` = recurrence quality

## Recommended defaults

- `alpha = 0.55`

Phase 1 quality contributors:

- seen across distinct sessions
- seen across distinct tasks
- seen across distinct days

Phase 1 noise contributors:

- same-session burst ratio
- same-actor concentration
- exact duplicate ratio
- too-small inter-arrival time

## Thresholds

Suggested initial thresholds:

- `S_select >= 0.72` -> strong candidate for later promotion review
- `0.45 <= S_select < 0.72` -> compression or summary candidate
- `S_select < 0.45` -> keep fast or expire by retention policy

These thresholds are product defaults, not protocol guarantees.
