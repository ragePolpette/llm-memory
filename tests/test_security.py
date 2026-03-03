"""Security tests: nessuna chiamata di rete outbound."""

from __future__ import annotations

import socket

import pytest

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
