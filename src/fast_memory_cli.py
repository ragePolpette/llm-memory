"""CLI locale per la distillazione agentica della fast memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .bootstrap import build_runtime
from .config import Config, get_config
from .service.memory_service import ActorContext


@dataclass
class PreparedDistillationArtifacts:
    output_dir: Path
    pack_path: Path
    prompt_path: Path
    prompt_text: str
    payload: dict


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _build_actor(args: argparse.Namespace, config: Config) -> ActorContext:
    return ActorContext(
        agent_id=args.agent_id,
        user_id=args.user_id,
        workspace_id=args.workspace_id or config.default_workspace_id,
        project_id=args.project_id or config.default_project_id,
    )


def _render_prompt(payload: dict) -> str:
    return (
        f"{payload['prompt']}\n\n"
        "Distillation contract:\n"
        f"{json.dumps(payload['contract'], ensure_ascii=True, indent=2)}\n\n"
        "Candidate pack:\n"
        f"{json.dumps(payload['candidates'], ensure_ascii=True, indent=2, default=str)}\n"
    )


def _default_output_dir(config: Config) -> Path:
    return config.import_export_base_dir / "distillation_runs" / _utc_stamp()


def _write_artifacts(
    *,
    config: Config,
    payload: dict,
    output_dir: Optional[Path],
) -> PreparedDistillationArtifacts:
    target_dir = (output_dir or _default_output_dir(config)).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = _render_prompt(payload)
    pack_path = target_dir / "candidate-pack.json"
    prompt_path = target_dir / "distillation-prompt.txt"

    pack_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    prompt_path.write_text(prompt_text, encoding="utf-8")

    return PreparedDistillationArtifacts(
        output_dir=target_dir,
        pack_path=pack_path,
        prompt_path=prompt_path,
        prompt_text=prompt_text,
        payload=payload,
    )


def _prepare_payload(args: argparse.Namespace) -> tuple[Config, PreparedDistillationArtifacts]:
    config = get_config()
    runtime = build_runtime(config)
    actor = _build_actor(args, config)
    payload = runtime.service.prepare_fast_distillation(
        actor=actor,
        reason=args.reason,
        cluster_id=args.cluster_id,
        entry_id=args.entry_id,
        top_k=args.top_k,
        include_resolved=args.include_resolved,
        distillation_status=args.distillation_status,
    )
    artifacts = _write_artifacts(
        config=config,
        payload=payload,
        output_dir=Path(args.output_dir).expanduser() if args.output_dir else None,
    )
    return config, artifacts


def _harness_command(args: argparse.Namespace) -> list[str]:
    executable = args.harness_bin or args.harness
    command = [executable]
    for extra_arg in args.harness_arg:
        command.append(extra_arg)
    return command


def _print_prepare_summary(artifacts: PreparedDistillationArtifacts) -> None:
    summary = {
        "prepared_count": artifacts.payload.get("prepared_count", 0),
        "output_dir": str(artifacts.output_dir),
        "pack_path": str(artifacts.pack_path),
        "prompt_path": str(artifacts.prompt_path),
        "protection": artifacts.payload.get("protection"),
    }
    sys.stdout.write(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")


def _run_harness(command: list[str], prompt_text: str) -> int:
    try:
        completed = subprocess.run(command, input=prompt_text, text=True, check=False)
    except FileNotFoundError:
        sys.stderr.write(f"Harness executable not found: {command[0]}\n")
        return 127
    return int(completed.returncode)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--user-id")
    parser.add_argument("--workspace-id")
    parser.add_argument("--project-id")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--cluster-id")
    parser.add_argument("--entry-id")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--include-resolved", action="store_true")
    parser.add_argument(
        "--distillation-status",
        choices=["pending", "summarized", "promoted", "discarded"],
    )
    parser.add_argument("--output-dir")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-memory-fast-distill",
        description="Prepara o lancia la distillazione agentica della fast memory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepara candidate pack e prompt fisso.")
    _add_common_arguments(prepare_parser)

    run_parser = subparsers.add_parser("run", help="Prepara pack+prompt e lancia l'harness scelto.")
    _add_common_arguments(run_parser)
    run_parser.add_argument("--harness", choices=["codex", "claude"], required=True)
    run_parser.add_argument("--harness-bin")
    run_parser.add_argument("--harness-arg", action="append", default=[])

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        _, artifacts = _prepare_payload(args)
    except PermissionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"Failed to prepare distillation payload: {exc}\n")
        return 1

    _print_prepare_summary(artifacts)
    if args.command == "prepare":
        return 0

    return _run_harness(_harness_command(args), artifacts.prompt_text)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
