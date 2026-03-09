from __future__ import annotations

import pytest

from src.embedding.embedding_service import EmbeddingProvider
from src.service.memory_service import ActorContext, MemoryInputError


class _FailingEmbeddingProvider(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed-fail")

    def dimension(self) -> int:
        return 4

    def provider_id(self) -> str:
        return "test-provider"

    def model_id(self) -> str:
        return "failing-embed"

    def fingerprint(self) -> str:
        return "failing-embed-fingerprint"


@pytest.mark.asyncio
async def test_add_degrades_when_embedding_fails(service):
    actor = ActorContext(agent_id="agent-nov-embed", user_id="user-nov-embed", workspace_id="ws-test", project_id="prj-test")
    service.embedding_provider = _FailingEmbeddingProvider()

    result = await service.add(
        {
            "content": "Il progetto usa SQLite locale come backend persistente del runtime.",
            "context": "architecture",
            "agent_id": actor.agent_id,
            "visibility": "shared",
        },
        actor,
    )

    assert result["success"] is True
    assert result["novelty_computed"] is False
    assert result["novelty_score"] is None

    entry = service.get(result["entry_id"], actor)
    assert entry is not None
    assert entry.metadata["novelty_status"] == "failed"
    assert entry.metadata["novelty_score"] is None

    audits = service.store.list_audit(limit=20)
    assert any(
        audit.action == "novelty_computation_failed"
        and audit.payload["error_type"] == "RuntimeError"
        and audit.payload["embedding_status"] == "failed"
        for audit in audits
    )


@pytest.mark.asyncio
async def test_add_degrades_when_semantic_dedup_fails(service):
    actor = ActorContext(agent_id="agent-nov-dedup", user_id="user-nov-dedup", workspace_id="ws-test", project_id="prj-test")

    await service.add(
        {
            "content": "Il progetto usa MemoryService come orchestratore della persistenza.",
            "context": "architecture",
            "agent_id": actor.agent_id,
            "visibility": "shared",
        },
        actor,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("semantic-fail")

    service.vector_store.similarity_search = _boom  # type: ignore[method-assign]

    result = await service.add(
        {
            "content": "Il progetto usa SQLite locale come backend persistente del runtime.",
            "context": "architecture",
            "agent_id": actor.agent_id,
            "visibility": "shared",
        },
        actor,
    )

    assert result["success"] is True
    assert result["novelty_score"] != 1.0

    audits = service.store.list_audit(limit=20)
    assert any(audit.action == "semantic_dedup_failed" for audit in audits)


@pytest.mark.asyncio
async def test_import_degrades_when_novelty_is_unavailable(service):
    actor = ActorContext(agent_id="agent-nov-import", user_id="user-nov-import", workspace_id="ws-test", project_id="prj-test")

    def _boom(*args, **kwargs):
        raise RuntimeError("search-fail")

    service.vector_store.search = _boom  # type: ignore[method-assign]

    jsonl_path = service.config.import_export_base_dir / "novelty-fail-import.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        '{"id":"e-nov-import","tier":"tier-2","scope":{"workspace_id":"ws-test","project_id":"prj-test"},"visibility":"shared","source":"test","type":"fact","status":"active","content":"Il sistema supporta import locali di memorie persistenti tramite JSONL.","context":"import","tags":[],"sensitivity_tags":[],"metadata":{},"links":[],"confidence":0.7,"created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","content_hash":"nov-import","embedding_version_id":null,"encrypted":false,"redacted":false}\n',
        encoding="utf-8",
    )

    result = await service.import_data(jsonl_path, "jsonl", actor)

    assert result.imported == 1

    entry = service.get("e-nov-import", actor)
    assert entry is not None
    assert entry.metadata["novelty_status"] == "failed"
    assert entry.metadata["novelty_score"] is None


@pytest.mark.asyncio
async def test_malformed_payload_does_not_create_novelty_score(service):
    actor = ActorContext(agent_id="agent-nov-invalid", user_id="user-nov-invalid", workspace_id="ws-test", project_id="prj-test")

    with pytest.raises(MemoryInputError):
        await service.add(
            {
                "content": "",
                "agent_id": actor.agent_id,
            },
            actor,
        )

    assert service.list_entries(actor, limit=10) == []
    audits = service.store.list_audit(limit=20)
    assert any(audit.action == "write_attempt" and audit.payload["outcome"] == "invalid" for audit in audits)
