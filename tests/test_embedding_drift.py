from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from src.embedding.embedding_service import HashEmbeddingProvider, SentenceTransformerProvider
from src.service.memory_service import ActorContext, EmbeddingVersionMismatchError

ACTOR = ActorContext(agent_id="agent-drift", user_id="user-drift", workspace_id="ws-test", project_id="prj-test")


async def _add(service, content: str) -> dict:
    result = await service.add(
        {"content": content, "context": "architecture", "agent_id": ACTOR.agent_id, "visibility": "shared"},
        ACTOR,
    )
    assert result["success"] is True
    return result


def test_sentence_transformer_fingerprint_is_stable_across_model_load():
    provider = SentenceTransformerProvider("sentence-transformers/test", dim_hint=384)
    before = provider.fingerprint()

    provider._model = object()
    provider._dimension = 768

    assert provider.fingerprint() == before
    assert provider.dimension() == 768


@pytest.mark.asyncio
async def test_provider_change_refuses_search_until_reembed(service):
    first = await _add(service, "Il progetto usa SQLite locale come backend persistente del runtime.")
    original_version = first["embedding_version_id"]

    service.embedding_provider = HashEmbeddingProvider(model_name="local-hash-other", dim=96)

    drift = service.embedding_version_drift()
    assert drift is not None
    assert drift["active_version_id"] == original_version

    with pytest.raises(EmbeddingVersionMismatchError, match="memory.reembed"):
        await service.search("backend persistente", ACTOR)

    second = await _add(service, "Il progetto mantiene audit trail locali per le operazioni di memoria.")
    assert second["embedding_version_id"] != original_version
    assert service.store.get_active_embedding_version().version_id == original_version

    result = await service.reembed(actor=ACTOR)

    assert result.version_id == second["embedding_version_id"]
    assert result.remaining == 0
    assert service.embedding_version_drift() is None
    bundles = await service.search("backend persistente", ACTOR, limit=5)
    assert {bundle.entry_id for bundle in bundles} >= {first["entry_id"]}


@pytest.mark.asyncio
async def test_prewarm_logs_explicit_warning_on_version_mismatch(runtime):
    await _add(runtime.service, "Il progetto usa SQLite locale come backend persistente del runtime.")
    runtime.service.embedding_provider = HashEmbeddingProvider(model_name="local-hash-other", dim=96)

    with capture_logs() as logs:
        await runtime.prewarm()

    warnings = [log for log in logs if log["event"] == "embedding_version_mismatch"]
    assert len(warnings) == 1
    assert warnings[0]["log_level"] == "warning"
    assert "Semantic search is disabled" in warnings[0]["message"]


@pytest.mark.asyncio
async def test_prewarm_is_silent_when_versions_match(runtime):
    await _add(runtime.service, "Il progetto usa SQLite locale come backend persistente del runtime.")

    with capture_logs() as logs:
        await runtime.prewarm()

    assert not [log for log in logs if log["event"] == "embedding_version_mismatch"]
    assert runtime.service.check_embedding_version() is None
