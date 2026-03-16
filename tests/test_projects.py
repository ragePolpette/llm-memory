from __future__ import annotations

import pytest

from src.service.memory_service import ActorContext


def test_default_project_is_bootstrapped(service, test_config):
    actor = ActorContext(
        agent_id="agent-a",
        user_id="user-a",
        workspace_id=test_config.default_workspace_id,
        project_id=test_config.default_project_id,
    )

    projects = service.list_projects(actor)

    assert len(projects) == 1
    assert projects[0].project_id == test_config.default_project_id


def test_create_project_registers_project_and_is_idempotent(service):
    actor = ActorContext(
        agent_id="agent-a",
        user_id="user-a",
        workspace_id="ws-test",
        project_id="prj-test",
    )

    first = service.create_project(
        actor=actor,
        project_id="project-alpha",
        display_name="Project Alpha",
        description="Alpha repo",
    )
    second = service.create_project(
        actor=actor,
        project_id="project-alpha",
        display_name="Ignored rename",
    )

    projects = service.list_projects(actor)

    assert first.project_id == "project-alpha"
    assert second.project_id == "project-alpha"
    assert service.get_project_info(actor, "project-alpha") is not None
    assert any(project.project_id == "project-alpha" for project in projects)


def test_create_project_rejects_invalid_identifier(service):
    actor = ActorContext(
        agent_id="agent-a",
        user_id="user-a",
        workspace_id="ws-test",
        project_id="prj-test",
    )

    with pytest.raises(ValueError, match="project_id must match"):
        service.create_project(actor=actor, project_id="Bad Project")
