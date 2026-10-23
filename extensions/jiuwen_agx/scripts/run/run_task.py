# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX task runner — the Windows entry point for a natural-language task.

Why this exists next to ``examples/run_task.py``: the generic runner goes
through ``jiuwensymbiosis.runtime.bindings`` for resource admission, whose
import chain reaches ``runtime/jobs.py`` -> ``import fcntl`` (Unix-only). On
Windows that fails before the robot is built. This runner follows the same
session-builder path as ``act_acceptance.py`` (``build_*.from_yaml``), so it
works on Windows without the admission layer.

It is machine-agnostic: ``--config`` selects the body (the YAML's ``adapter:``
field), so the same command runs 365 and 215F. Relative paths in ``agent:``
(``workspace`` / ``episode_dir`` / ``trace_dir`` / ``log_dir``) are resolved
against the repository root, not the current directory.

Runtime output (per the 215F/365 YAMLs):

    extensions/jiuwen_agx/runtime/
      agent/       # openjiuwen workspace scaffolding (AGENT.md, memory/, ...)
      episodes/    # planning-level *.jsonl
      traces/      # tool-level *.json + frames/
      logs/
        jiuwensymbiosis.log
        openjiuwen/{run,interface,performance}/

Usage::

    .venv\\Scripts\\python.exe extensions\\jiuwen_agx\\scripts\\run\\run_task.py ^
        --config extensions\\jiuwen_agx\\configs\\agx_excavator\\agx_excavator.215f.local.yaml ^
        --query "把土堆挖一斗倒到旁边"

Use ``examples/run_task.py`` instead when you need its extra surfaces (mock
model, voice, resource admission) and are not on Windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EXT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | None) -> str | None:
    """Resolve an ``agent:`` path against the repository root."""
    if not value:
        return value
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return str(path.resolve())


def _clear_proxy_env_early() -> None:
    """Clear proxy vars BEFORE importing openjiuwen (AGENTS.md proxy hygiene).

    Mirrors the inline block at the top of ``jiuwensymbiosis/__init__.py``: the
    package import would clear them too, but that import also pulls openjiuwen,
    and openjiuwen's log backend needs to be configured first. Keeping the
    popped proxy in ``JIUWEN_LLM_PROXY`` preserves cloud-LLM access.
    """
    popped: dict[str, str] = {}
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        value = os.environ.pop(key, None)
        if value is not None:
            popped[key] = value
    proxy = (
        popped.get("https_proxy")
        or popped.get("HTTPS_PROXY")
        or popped.get("http_proxy")
        or popped.get("HTTP_PROXY")
        or popped.get("all_proxy")
        or popped.get("ALL_PROXY")
    )
    if proxy:
        os.environ.setdefault("JIUWEN_LLM_PROXY", proxy)
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")


def _prepare_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the YAML and normalise its ``agent:`` paths (absolute)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    agent_raw = dict(raw.get("agent") or {})
    for key in ("workspace", "episode_dir", "trace_dir", "log_dir"):
        if key in agent_raw:
            agent_raw[key] = _resolve(agent_raw[key])
    return raw, agent_raw


def _configure_openjiuwen_logging(agent_raw: dict[str, Any], *, console: bool) -> None:
    """Route openjiuwen's own log backend into the managed runtime directory.

    Its default is ``log_path: ./logs/`` (cwd-relative, and it ends up under
    ``logs/logs/``). An absolute path here produces ``run/``, ``interface/`` and
    ``performance/`` directly under the chosen directory. Best-effort: a logging
    setup failure must not stop the task.
    """
    log_dir = Path(agent_raw.get("log_dir") or (_EXT_ROOT / "runtime" / "logs"))
    if not log_dir.is_absolute():
        log_dir = _REPO_ROOT / log_dir
    oj_log_dir = log_dir / "openjiuwen"
    try:
        from openjiuwen.core.common.logging.log_config import configure_log_config

        configure_log_config(
            {
                "backend": "default",
                "level": str(agent_raw.get("log_level", "INFO")),
                "log_path": str(oj_log_dir.resolve()) + os.sep,
                "log_file": "run/jiuwen.log",
                # console=True keeps openjiuwen's ~20 init lines visible for
                # debugging; the default keeps the terminal to framework logs.
                "output": ["console", "file"] if console else ["file"],
            }
        )
    except Exception as exc:  # noqa: BLE001 - logging must not block the task
        print(
            f"warning: could not configure openjiuwen logging: {exc}", file=sys.stderr
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        required=True,
        help="Robot config YAML (its adapter: field picks the body)",
    )
    parser.add_argument("--query", required=True, help="Natural-language task")
    parser.add_argument(
        "--openjiuwen-console",
        action="store_true",
        help="Also print openjiuwen's own logs to the terminal (default: file only).",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    raw, agent_raw = _prepare_config(config_path)

    _clear_proxy_env_early()
    # openjiuwen's log backend must be configured before jiuwensymbiosis imports it.
    _configure_openjiuwen_logging(agent_raw, console=args.openjiuwen_console)

    from jiuwensymbiosis import run_robot_task
    from jiuwensymbiosis._extensions import discover_adapter_builders
    from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig
    from jiuwensymbiosis.utils.logging import configure_logging

    # One owned StreamHandler + <log_dir>/jiuwensymbiosis.log. build_robot_agent
    # calls this again with the same values; configure_logging is idempotent.
    configure_logging(
        level=str(agent_raw.get("log_level", "INFO")), log_dir=agent_raw.get("log_dir")
    )

    model_raw = dict(raw.get("model") or {})
    spec = ModelSpec(
        provider=str(model_raw.get("provider", "OpenAI")),
        api_base=str(model_raw.get("api_base", "")),
        api_key=str(model_raw.get("api_key", "")),
        model_name=str(model_raw.get("model_name", "")),
        temperature=float(model_raw.get("temperature", 0.3)),
        max_tokens=int(model_raw.get("max_tokens", 2048)),
        verify_ssl=False,
    )
    agent_cfg = RobotAgentConfig.from_dict(agent_raw)
    agent_cfg.model_spec = spec

    adapter = str(raw.get("adapter") or "agx_excavator")
    builders = discover_adapter_builders()
    if adapter not in builders:
        parser.error(f"unknown adapter {adapter!r}; available: {sorted(builders)}")
    session = builders[adapter].from_yaml(str(config_path))
    with session:
        result = run_robot_task(session, args.query, agent_cfg)

    print("=== Agent result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=repr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
