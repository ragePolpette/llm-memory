from __future__ import annotations

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
