from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from multi_agent.environment import collect_environment
from multi_agent.metrics import TOKEN_COUNT_METHOD
from multi_agent.runner import ExperimentRunner
from multi_agent.tasks import DEFAULT_TASKS


VARIANTS = [
    {
        "name": "text_baseline",
        "mode": "text",
        "description": "传统自然语言交接，不使用共享记忆检索、状态引用或记忆网络。",
        "features": {
            "FEATURE_MEMORY_SEARCH": "0",
            "FEATURE_MEMORY_WRITE": "0",
            "FEATURE_STATE_EXCHANGE": "0",
            "MEMORY_GRAPH_ENABLED": "0",
        },
    },
    {
        "name": "structured_protocol",
        "mode": "structured",
        "description": "仅启用结构化 Agent-to-Agent 协议和能力握手。",
        "features": {
            "FEATURE_MEMORY_SEARCH": "0",
            "FEATURE_MEMORY_WRITE": "0",
            "FEATURE_STATE_EXCHANGE": "0",
            "MEMORY_GRAPH_ENABLED": "0",
        },
    },
    {
        "name": "structured_state",
        "mode": "structured",
        "description": "结构化协议 + StateStore，embedding/tool result 通过 state_id 传递。",
        "features": {
            "FEATURE_MEMORY_SEARCH": "0",
            "FEATURE_MEMORY_WRITE": "0",
            "FEATURE_STATE_EXCHANGE": "1",
            "MEMORY_GRAPH_ENABLED": "0",
        },
    },
    {
        "name": "structured_memory",
        "mode": "structured",
        "description": "结构化协议 + StateStore + 基础共享记忆，使用 embedding/tag/text 混合检索。",
        "features": {
            "FEATURE_MEMORY_SEARCH": "1",
            "FEATURE_MEMORY_WRITE": "1",
            "FEATURE_STATE_EXCHANGE": "1",
            "MEMORY_GRAPH_ENABLED": "0",
        },
    },
    {
        "name": "structured_memory_graph",
        "mode": "structured",
        "description": "完整系统：结构化协议 + 状态引用 + 共享记忆 + keywords/links 动态记忆网络。",
        "features": {
            "FEATURE_MEMORY_SEARCH": "1",
            "FEATURE_MEMORY_WRITE": "1",
            "FEATURE_STATE_EXCHANGE": "1",
            "MEMORY_GRAPH_ENABLED": "1",
        },
    },
]


def main() -> None:
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir = Path(os.getenv("ABLATION_DATA_DIR", "data/ablation"))
    ablation_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    rounds = int(os.getenv("ABLATION_ROUNDS", os.getenv("EXPERIMENT_ROUNDS", "1")))
    task_limit = int(os.getenv("ABLATION_TASK_LIMIT", "0"))
    tasks = DEFAULT_TASKS[:task_limit] if task_limit > 0 else DEFAULT_TASKS

    started = time.perf_counter()
    variant_results = []
    for variant in VARIANTS:
        variant_results.append(_run_variant(variant, tasks, rounds, run_id, ablation_dir))

    results = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rounds": rounds,
        "tasks": tasks,
        "environment": collect_environment(),
        "variants": variant_results,
        "comparison": _compare_variants(variant_results),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }

    json_path = output_dir / "ablation_results.json"
    report_path = output_dir / "ablation_report.txt"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(results), encoding="utf-8")

    print(json.dumps(results["comparison"], ensure_ascii=False, indent=2))
    print(f"\nAblation JSON written to {json_path}")
    print(f"Ablation report written to {report_path}")


def _run_variant(
    variant: dict[str, Any],
    tasks: list[str],
    rounds: int,
    run_id: str,
    ablation_dir: Path,
) -> dict[str, Any]:
    name = variant["name"]
    env = {
        **variant["features"],
        "MEMORY_PATH": str(ablation_dir / f"{run_id}_{name}_memory.sqlite"),
        "STATE_PATH": str(ablation_dir / f"{run_id}_{name}_state.sqlite"),
        "MEMORY_RESET": "1",
        "STATE_RESET": "1",
    }
    runs = []
    with _temporary_env(env):
        runner = ExperimentRunner()
        for round_index in range(rounds):
            for task in tasks:
                run = runner.run_task(task=task, mode=variant["mode"])
                run["round"] = round_index + 1
                run["variant"] = name
                runs.append(run)
        memory = runner.memory.to_dict()
        states = runner.state_store.to_dict()

    return {
        "name": name,
        "mode": variant["mode"],
        "description": variant["description"],
        "features": variant["features"],
        "summary": _summarize(runs, memory, states),
        "runs": runs,
        "memory": memory,
        "states": states,
    }


def _summarize(runs: list[dict[str, Any]], memory: list[dict[str, Any]], states: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [run["metrics"] for run in runs]
    return {
        "runs": len(runs),
        "token_count_method": TOKEN_COUNT_METHOD,
        "avg_messages": _avg(metrics, "message_count"),
        "avg_chars": _avg(metrics, "char_count"),
        "std_chars": _std(metrics, "char_count"),
        "avg_approx_tokens": _avg(metrics, "approx_token_count"),
        "std_approx_tokens": _std(metrics, "approx_token_count"),
        "avg_elapsed_ms": _avg(metrics, "elapsed_ms"),
        "std_elapsed_ms": _std(metrics, "elapsed_ms"),
        "avg_memory_hits": _avg(metrics, "memory_hit_count"),
        "avg_non_text_transfers": _avg(metrics, "non_text_transfer_count"),
        "avg_non_text_transfer_size": _avg(metrics, "non_text_transfer_size"),
        "avg_protocol_events": _avg(metrics, "protocol_event_count"),
        "avg_protocol_approx_tokens": _avg(metrics, "protocol_approx_token_count"),
        "memory_records": len(memory),
        "state_records": len(states),
        "avg_memory_links": _avg_memory_links(memory),
        "memory_access_count": sum(int(item.get("access_count", 0) or 0) for item in memory),
    }


def _compare_variants(variants: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {variant["name"]: variant["summary"] for variant in variants}
    baseline = by_name["text_baseline"]
    comparison = {}
    for name, summary in by_name.items():
        comparison[name] = {
            "token_delta_vs_text_pct": _pct_delta(summary["avg_approx_tokens"], baseline["avg_approx_tokens"]),
            "char_delta_vs_text_pct": _pct_delta(summary["avg_chars"], baseline["avg_chars"]),
            "elapsed_delta_vs_text_pct": _pct_delta(summary["avg_elapsed_ms"], baseline["avg_elapsed_ms"]),
            "memory_hit_delta_vs_text": round(summary["avg_memory_hits"] - baseline["avg_memory_hits"], 3),
            "state_transfer_delta_vs_text": round(
                summary["avg_non_text_transfers"] - baseline["avg_non_text_transfers"], 3
            ),
        }

    ordered_names = [variant["name"] for variant in variants]
    marginal = {}
    for previous, current in zip(ordered_names, ordered_names[1:]):
        previous_summary = by_name[previous]
        current_summary = by_name[current]
        marginal[f"{previous}_to_{current}"] = {
            "token_delta_pct": _pct_delta(
                current_summary["avg_approx_tokens"], previous_summary["avg_approx_tokens"]
            ),
            "elapsed_delta_pct": _pct_delta(current_summary["avg_elapsed_ms"], previous_summary["avg_elapsed_ms"]),
            "memory_hit_delta": round(
                current_summary["avg_memory_hits"] - previous_summary["avg_memory_hits"], 3
            ),
            "state_records_delta": current_summary["state_records"] - previous_summary["state_records"],
            "memory_links_delta": round(
                current_summary["avg_memory_links"] - previous_summary["avg_memory_links"], 3
            ),
        }
    return {"vs_text": comparison, "marginal": marginal}


def _render_report(results: dict[str, Any]) -> str:
    environment = results.get("environment", {})
    lines = [
        "# Multi-Agent Mechanism Ablation Report",
        "",
        f"- Generated at: {results['generated_at']}",
        f"- Run ID: {results['run_id']}",
        f"- Rounds: {results['rounds']}",
        f"- Tasks per round: {len(results['tasks'])}",
        f"- OS: {environment.get('os_pretty_name', 'unknown')}",
        f"- LLM backend: {environment.get('llm_backend', 'required')}",
        f"- LLM model: {environment.get('llm_model', 'required')}",
        f"- Embedding backend: {environment.get('embedding_backend', 'hash')}",
        f"- Token count method: {TOKEN_COUNT_METHOD}",
        f"- State backend: {environment.get('state_backend', 'shared_memory')}",
        f"- CodeAct sandbox: {environment.get('codeact_sandbox', 'subprocess')}",
        "",
        "## Variant Definitions",
        "",
        "| Variant | Mode | Enabled Mechanisms |",
        "| --- | --- | --- |",
    ]
    for variant in results["variants"]:
        enabled = _enabled_features(variant["features"])
        lines.append(f"| {variant['name']} | {variant['mode']} | {enabled} |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Variant | Avg Tokens | Token Delta | Avg Chars | Char Delta | Avg Elapsed ms | Memory Hits | State Transfers | Memory Records | State Records | Avg Links |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    comparison = results["comparison"]["vs_text"]
    for variant in results["variants"]:
        summary = variant["summary"]
        token_delta = comparison[variant["name"]]["token_delta_vs_text_pct"]
        char_delta = comparison[variant["name"]]["char_delta_vs_text_pct"]
        lines.append(
            "| {name} | {tokens} | {token_delta} | {chars} | {char_delta} | {elapsed} | {hits} | {transfers} | {memories} | {states} | {links} |".format(
                name=variant["name"],
                tokens=summary["avg_approx_tokens"],
                token_delta=_fmt_pct(token_delta),
                chars=summary["avg_chars"],
                char_delta=_fmt_pct(char_delta),
                elapsed=summary["avg_elapsed_ms"],
                hits=summary["avg_memory_hits"],
                transfers=summary["avg_non_text_transfers"],
                memories=summary["memory_records"],
                states=summary["state_records"],
                links=summary["avg_memory_links"],
            )
        )

    lines.extend(
        [
            "",
            "## Marginal Contribution",
            "",
            "| Step | Token Delta | Elapsed Delta | Memory Hit Delta | State Record Delta | Memory Link Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for step, values in results["comparison"]["marginal"].items():
        lines.append(
            "| {step} | {token} | {elapsed} | {hits} | {states} | {links} |".format(
                step=step,
                token=_fmt_pct(values["token_delta_pct"]),
                elapsed=_fmt_pct(values["elapsed_delta_pct"]),
                hits=values["memory_hit_delta"],
                states=values["state_records_delta"],
                links=values["memory_links_delta"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `structured_protocol` isolates the benefit of compact protocol encoding.",
            "- `structured_state` shows the extra observability and artifact reuse enabled by non-text state references.",
            "- `structured_memory` measures cross-task reuse from persistent shared memory.",
            "- `structured_memory_graph` adds dynamic keywords and links, following agentic-memory ideas such as Zettelkasten-style organization.",
            "",
            "Formal submission should rerun this report with `ABLATION_ROUNDS=10`, real embedding, and the target openEuler 24.03-LTS-SP3 environment.",
        ]
    )
    return "\n".join(lines) + "\n"


def _avg(metrics: list[dict[str, Any]], key: str) -> float:
    return round(mean(float(item[key]) for item in metrics), 3) if metrics else 0.0


def _std(metrics: list[dict[str, Any]], key: str) -> float:
    return round(pstdev(float(item[key]) for item in metrics), 3) if len(metrics) > 1 else 0.0


def _avg_memory_links(memory: list[dict[str, Any]]) -> float:
    if not memory:
        return 0.0
    return round(sum(len(item.get("links", [])) for item in memory) / len(memory), 3)


def _pct_delta(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value}%"


def _enabled_features(features: dict[str, str]) -> str:
    names = []
    if features.get("FEATURE_STATE_EXCHANGE") == "1":
        names.append("state refs")
    if features.get("FEATURE_MEMORY_SEARCH") == "1" or features.get("FEATURE_MEMORY_WRITE") == "1":
        names.append("shared memory")
    if features.get("MEMORY_GRAPH_ENABLED") == "1":
        names.append("memory graph")
    return ", ".join(names) or "protocol/text only"


@contextmanager
def _temporary_env(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
