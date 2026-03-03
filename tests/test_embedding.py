"""Unit test: embedding provider e versioning."""

from __future__ import annotations

import pytest

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
