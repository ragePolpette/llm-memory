"""Unit test: embedding provider e versioning."""

from __future__ import annotations

import asyncio

import pytest

from src.embedding.embedding_service import EmbeddingProvider, SentenceTransformerProvider
from src.models import ScopeRef
from src.service.memory_service import ActorContext


@pytest.mark.asyncio
async def test_embedding_versioning_and_reembed(service):
    actor = ActorContext(agent_id="agent-embed", user_id="user-embed", workspace_id="ws-test", project_id="prj-test")

    await service.add(
        {
            "content": "Decisione: usare storage locale e niente cloud.",
            "context": "decision",
            "agent_id": actor.agent_id,
            "tier": "tier-2",
            "type": "decision",
            "visibility": "shared",
        },
        actor,
    )

    versions_before = service.store.list_embedding_versions()
    assert len(versions_before) >= 1

    reembed_result = await service.reembed(actor=actor, model_id="local-hash-test-v2", activate=True)
    assert reembed_result.processed >= 1
    assert reembed_result.remaining == 0

    versions_after = service.store.list_embedding_versions()
    assert len(versions_after) >= len(versions_before)
    assert any(v.version_id == reembed_result.version_id for v in versions_after)

    active = service.store.get_active_embedding_version()
    assert active is not None
    assert active.version_id == reembed_result.version_id


@pytest.mark.asyncio
async def test_reembed_uses_override_dimension_for_generated_vectors(service):
    actor = ActorContext(agent_id="agent-embed", user_id="user-embed", workspace_id="ws-test", project_id="prj-test")

    add_result = await service.add(
        {
            "content": "Decisione: usare storage locale e niente cloud.",
            "context": "decision",
            "agent_id": actor.agent_id,
            "tier": "tier-2",
            "type": "decision",
            "visibility": "shared",
        },
        actor,
    )

    reembed_result = await service.reembed(actor=actor, dim=32, activate=True)

    active = service.store.get_active_embedding_version()
    assert active is not None
    assert active.version_id == reembed_result.version_id
    assert active.dim == 32

    vector = service.store.get_embedding(add_result["entry_id"], reembed_result.version_id)
    assert vector is not None
    assert len(vector) == 32


class _PartiallyFailingEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        self._dim = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            if "fail-vector" in text:
                result.append([0.0, 0.0, 0.0, 0.0])
            else:
                result.append([1.0, 0.0, 0.0, 0.0])
        return result

    def dimension(self) -> int:
        return self._dim

    def provider_id(self) -> str:
        return "test-provider"

    def model_id(self) -> str:
        return "partial-fail"

    def fingerprint(self) -> str:
        return "partial-fail-fingerprint"


@pytest.mark.asyncio
async def test_reembed_skips_permanent_failures_without_looping(service):
    actor = ActorContext(agent_id="agent-embed", user_id="user-embed", workspace_id="ws-test", project_id="prj-test")

    await service.add(
        {
            "content": "fail-vector content",
            "context": "decision",
            "agent_id": actor.agent_id,
            "tier": "tier-2",
            "type": "decision",
            "visibility": "shared",
        },
        actor,
    )
    success = await service.add(
        {
            "content": "good-vector content",
            "context": "decision",
            "agent_id": actor.agent_id,
            "tier": "tier-2",
            "type": "decision",
            "visibility": "shared",
        },
        actor,
    )

    service.embedding_provider = _PartiallyFailingEmbeddingProvider()

    result = await service.reembed(actor=actor, model_id="partial-fail", activate=True, batch_size=1)

    assert result.processed == 1
    assert result.skipped == 1
    assert result.remaining == 1

    scope = ScopeRef(workspace_id=actor.workspace_id, project_id=actor.project_id)
    embeddings = service.store.list_embeddings(result.version_id, scope=scope, include_invalidated=True)
    embedded_ids = {entry.id for entry, _ in embeddings}
    assert success["entry_id"] in embedded_ids


def test_sentence_transformer_dimension_uses_hint_without_loading(monkeypatch):
    provider = SentenceTransformerProvider("sentence-transformers/test", dim_hint=321)

    def fail_load():
        raise AssertionError("dimension() should not trigger model loading")

    monkeypatch.setattr(provider, "_load_model", fail_load)

    assert provider.dimension() == 321
    assert provider.fingerprint()


@pytest.mark.asyncio
async def test_sentence_transformer_prepare_loads_model_via_to_thread(monkeypatch):
    provider = SentenceTransformerProvider("sentence-transformers/test", dim_hint=123)
    calls: list[str] = []

    def fake_load_model():
        calls.append("load")
        provider._model = object()
        provider._dimension = 456

    async def fake_to_thread(func, *args, **kwargs):
        calls.append("to_thread")
        return func(*args, **kwargs)

    monkeypatch.setattr(provider, "_load_model", fake_load_model)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    await provider.prepare()

    assert calls == ["to_thread", "load"]
    assert provider.dimension() == 456
