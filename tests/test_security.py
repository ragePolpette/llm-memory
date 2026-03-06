"""Security tests: nessuna chiamata di rete outbound."""

from __future__ import annotations

import asyncio
import socket

import pytest

from src.config import MemoryScope, Tier
from src.models import EntryType, MemoryEntry, ScopeRef
from src.security.no_network import NetworkBlockedError
from src.service.memory_service import ActorContext


@pytest.mark.asyncio
async def test_network_guard_blocks_external(service):
    # build_runtime (fixture) installa network guard.
    with pytest.raises(NetworkBlockedError):
        socket.create_connection(("8.8.8.8", 53), timeout=1)

    actor = ActorContext(agent_id="agent-sec", user_id="user-sec", workspace_id="ws-test", project_id="prj-test")
    result = await service.add(
        {
            "content": "Dato locale senza rete",
            "context": "security",
            "agent_id": actor.agent_id,
            "tier": "tier-1",
            "type": "fact",
            "visibility": "shared",
        },
        actor,
    )
    assert result["success"] is True


@pytest.mark.asyncio
async def test_network_guard_blocks_asyncio_open_connection(service):
    with pytest.raises(NetworkBlockedError):
        await asyncio.open_connection("8.8.8.8", 53)


@pytest.mark.asyncio
async def test_network_guard_blocks_event_loop_create_connection(service):
    loop = asyncio.get_running_loop()

    with pytest.raises(NetworkBlockedError):
        await loop.create_connection(asyncio.Protocol, "8.8.8.8", 53)


@pytest.mark.asyncio
async def test_private_entry_requires_exact_agent_match(service):
    actor = ActorContext(agent_id="agent-owner", user_id="user-owner", workspace_id="ws-test", project_id="prj-test")

    result = await service.add(
        {
            "content": "Memoria privata vincolata all'agente.",
            "context": "security",
            "agent_id": actor.agent_id,
            "tier": "tier-1",
            "type": "fact",
            "visibility": "private",
        },
        actor,
    )

    entry = service.get(result["entry_id"], actor)
    assert entry is not None


def test_private_entry_without_agent_id_is_not_globally_readable(service):
    actor = ActorContext(agent_id="agent-reader", user_id="user-owner", workspace_id="ws-test", project_id="prj-test")

    entry = MemoryEntry(
        tier=Tier.TIER_1,
        scope=ScopeRef(
            workspace_id=actor.workspace_id,
            project_id=actor.project_id,
            user_id=actor.user_id,
            agent_id=None,
        ),
        visibility=MemoryScope.PRIVATE,
        type=EntryType.FACT,
        content="Legacy private entry without agent id",
        context="security",
    )
    service.store.add_entry(entry)

    with pytest.raises(PermissionError, match="Read denied by scope policy"):
        service.get(entry.id, actor)
