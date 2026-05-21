"""CLI entry point for the AI job-ad annotation pipeline.

Wraps the netrun engine to load the annotation-specific DAG
(`config/netrun_annotations.json`) and run definitions
(`config/run_defs_annotations.toml`), independent of the exposure pipeline
defined in `config/netrun.json`.

Usage:
    uv run run-ai-ad-annotations <run_name>

Falls back to the RUN_NAME env var if no argument is given.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from netrun.core import Net, NetConfig
from netrun.logging._backends import JsonlEpochLogger, JsonlNetActionLogger

from ai_index.const import config_path, logs_path
from ai_index.run_pipeline import _resolve_run_defs


NETRUN_CONFIG_PATH = config_path / "netrun_annotations.json"
RUN_DEFS_PATH = config_path / "run_defs_annotations.toml"


def _load_run_defs(run_defs_path: Path) -> dict:
    with open(run_defs_path, "rb") as f:
        return tomllib.load(f)


async def run_annotations_async(run_name: str) -> dict:
    """Load the annotation DAG and run definitions, then run the pipeline."""
    load_dotenv()

    run_defs = _load_run_defs(RUN_DEFS_PATH)
    global_vars, node_vars = _resolve_run_defs(run_defs, run_name)

    config = NetConfig.from_file(
        str(NETRUN_CONFIG_PATH),
        global_node_vars=global_vars,
        node_vars=node_vars,
    )
    config.project_root_override = str(Path.cwd())
    print(f"run-ai-ad-annotations: using DAG {NETRUN_CONFIG_PATH.name}, run definition {run_name!r}", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_logs_path = logs_path / "annotations" / run_name
    run_logs_path.mkdir(parents=True, exist_ok=True)
    epoch_log_file = run_logs_path / f"epochs_{ts}.jsonl"
    actions_log_file = run_logs_path / f"actions_{ts}.jsonl"

    with JsonlEpochLogger(epoch_log_file) as epoch_logger, \
         JsonlNetActionLogger(actions_log_file) as action_logger:
        async with Net(config) as net:
            net.on_epoch_end(epoch_logger)
            net.on_net_actions(action_logger)

            made_progress = True
            while made_progress:
                made_progress, _, _ = await net.run_until_blocked()

            results = net.flush_all_output_queues()
            for queue_name, outputs in results.items():
                print(f"\n=== Output queue: {queue_name} ({len(outputs)} packet(s)) ===")
                for i, output in enumerate(outputs):
                    print(f"  [{i}] {output}")

    print(f"run-ai-ad-annotations: logs saved to {run_logs_path}", flush=True)
    return results


def main():
    """Sync entry point: ``run-ai-ad-annotations <run_name>``."""
    args = sys.argv[1:]
    if len(args) > 1:
        print("Usage: run-ai-ad-annotations <RUN_NAME>", file=sys.stderr)
        sys.exit(1)
    run_name = args[0] if args else os.environ.get("RUN_NAME")
    if run_name is None:
        print(
            "run-ai-ad-annotations: run_name is required (positional arg or RUN_NAME env var)",
            file=sys.stderr,
        )
        sys.exit(1)
    asyncio.run(run_annotations_async(run_name))
