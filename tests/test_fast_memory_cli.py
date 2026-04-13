from __future__ import annotations

import json
from pathlib import Path

from src import fast_memory_cli
from src.service.memory_service import ActorContext


def test_prepare_command_writes_candidate_pack_and_prompt(runtime, monkeypatch, tmp_path: Path, capsys):
    runtime.service.config.fast_memory_agent_distillation_enabled = True
    runtime.service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x.",
            "context": "menu troubleshooting",
            "agent_id": "cli-agent",
            "event_type": "incident",
            "kind": "bug",
            "product_area": "authorization",
            "component": "menu-engine",
            "entity_refs": ["user:y", "menu:x"],
            "action_taken": "Update tabella permessi utente.",
            "outcome": "Menu ripristinati.",
            "generalizable": "yes",
            "recurrence_count": 2,
        },
        actor=ActorContext(
            agent_id="cli-agent",
            user_id="cli-user",
            workspace_id="ws-test",
            project_id="prj-test",
        ),
    )

    monkeypatch.setattr(fast_memory_cli, "get_config", lambda: runtime.config)
    monkeypatch.setattr(fast_memory_cli, "build_runtime", lambda config: runtime)

    exit_code = fast_memory_cli.main(
        [
            "prepare",
            "--agent-id",
            "cli-agent",
            "--reason",
            "prepare top candidate",
            "--top-k",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "candidate-pack.json").exists()
    assert (tmp_path / "distillation-prompt.txt").exists()
    stdout = capsys.readouterr().out
    assert "candidate-pack.json" in stdout


def test_run_command_invokes_selected_harness(runtime, monkeypatch, tmp_path: Path):
    runtime.service.config.fast_memory_agent_distillation_enabled = True
    runtime.service.log_fast(
        {
            "content": "Errore ricorrente nell'import incrementale dopo resume.",
            "agent_id": "cli-agent",
            "event_type": "incident",
            "component": "import-engine",
            "recurrence_count": 3,
        },
        actor=ActorContext(
            agent_id="cli-agent",
            user_id="cli-user",
            workspace_id="ws-test",
            project_id="prj-test",
        ),
    )

    monkeypatch.setattr(fast_memory_cli, "get_config", lambda: runtime.config)
    monkeypatch.setattr(fast_memory_cli, "build_runtime", lambda config: runtime)

    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0

    def _fake_run(command, input, text, check):
        captured["command"] = command
        captured["input"] = input
        captured["text"] = text
        captured["check"] = check
        return _Completed()

    monkeypatch.setattr(fast_memory_cli.subprocess, "run", _fake_run)

    exit_code = fast_memory_cli.main(
        [
            "run",
            "--agent-id",
            "cli-agent",
            "--reason",
            "distill top candidate",
            "--top-k",
            "1",
            "--harness",
            "codex",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["command"] == ["codex"]
    assert "Candidate pack:" in str(captured["input"])
    assert (tmp_path / "candidate-pack.json").exists()


def test_apply_command_dry_run_by_default(runtime, monkeypatch, tmp_path: Path, capsys):
    runtime.service.config.fast_memory_agent_distillation_apply_enabled = True
    actor = ActorContext(
        agent_id="cli-agent",
        user_id="cli-user",
        workspace_id="ws-test",
        project_id="prj-test",
    )
    first = runtime.service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor=actor,
    )
    second = runtime.service.log_fast(
        {
            "content": "L'utente z vedeva solo il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor=actor,
    )

    payload_path = tmp_path / "result.json"
    payload_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "cluster_id": "cluster-menu",
                        "action": "promote",
                        "summary": "La gestione menu dipende dai permessi utente consolidati.",
                        "confidence": 0.9,
                        "source_entry_ids": [first["entry_id"], second["entry_id"]],
                        "strong_memory": {
                            "content": "La gestione menu dipende dal corretto allineamento dei permessi utente.",
                            "context": "menu troubleshooting",
                            "type": "fact",
                            "tier": "tier-2",
                            "visibility": "shared",
                            "tags": ["menu", "permissions"],
                            "metadata": {"kind": "distilled-knowledge"},
                        },
                    }
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(fast_memory_cli, "get_config", lambda: runtime.config)
    monkeypatch.setattr(fast_memory_cli, "build_runtime", lambda config: runtime)

    exit_code = fast_memory_cli.main(
        [
            "apply",
            "--agent-id",
            "cli-agent",
            "--reason",
            "preview distillation result",
            "--input",
            str(payload_path),
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["success"] is True
    assert stdout["dry_run"] is True
    assert runtime.service.get_fast(first["entry_id"], actor).distillation_status.value == "pending"


def test_apply_command_applies_when_requested(runtime, monkeypatch, tmp_path: Path, capsys):
    runtime.service.config.fast_memory_agent_distillation_apply_enabled = True
    actor = ActorContext(
        agent_id="cli-agent",
        user_id="cli-user",
        workspace_id="ws-test",
        project_id="prj-test",
    )
    first = runtime.service.log_fast(
        {
            "content": "L'utente y vedeva solo il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor=actor,
    )
    second = runtime.service.log_fast(
        {
            "content": "L'utente z vedeva solo il menu x.",
            "agent_id": actor.agent_id,
            "event_type": "incident",
            "component": "menu-engine",
        },
        actor=actor,
    )

    payload_path = tmp_path / "result.json"
    payload_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "cluster_id": "cluster-menu",
                        "action": "promote",
                        "summary": "La gestione menu dipende dai permessi utente consolidati.",
                        "confidence": 0.9,
                        "source_entry_ids": [first["entry_id"], second["entry_id"]],
                        "strong_memory": {
                            "content": "La gestione menu dipende dal corretto allineamento dei permessi utente.",
                            "context": "menu troubleshooting",
                            "type": "fact",
                            "tier": "tier-2",
                            "visibility": "shared",
                            "tags": ["menu", "permissions"],
                            "metadata": {"kind": "distilled-knowledge"},
                        },
                    }
                ]
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(fast_memory_cli, "get_config", lambda: runtime.config)
    monkeypatch.setattr(fast_memory_cli, "build_runtime", lambda config: runtime)

    exit_code = fast_memory_cli.main(
        [
            "apply",
            "--agent-id",
            "cli-agent",
            "--reason",
            "apply distillation result",
            "--input",
            str(payload_path),
            "--apply",
        ]
    )

    assert exit_code == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["success"] is True
    assert stdout["dry_run"] is False
    assert stdout["results"][0]["promoted_entry_id"]
    assert runtime.service.get_fast(first["entry_id"], actor).distillation_status.value == "promoted"
    assert runtime.service.get_fast(second["entry_id"], actor).distillation_status.value == "summarized"


def test_apply_command_rejects_invalid_json(runtime, monkeypatch, tmp_path: Path, capsys):
    runtime.service.config.fast_memory_agent_distillation_apply_enabled = True
    payload_path = tmp_path / "result.json"
    payload_path.write_text("{invalid", encoding="utf-8")

    monkeypatch.setattr(fast_memory_cli, "get_config", lambda: runtime.config)
    monkeypatch.setattr(fast_memory_cli, "build_runtime", lambda config: runtime)

    exit_code = fast_memory_cli.main(
        [
            "apply",
            "--agent-id",
            "cli-agent",
            "--reason",
            "apply distillation result",
            "--input",
            str(payload_path),
        ]
    )

    assert exit_code == 2
    assert "not valid JSON" in capsys.readouterr().err
