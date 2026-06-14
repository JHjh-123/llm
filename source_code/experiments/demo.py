from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from multi_agent.runner import ExperimentRunner
from multi_agent.tasks import DEFAULT_TASKS


def main() -> None:
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    rounds = int(os.getenv("DEMO_ROUNDS", os.getenv("EXPERIMENT_ROUNDS", "10")))
    runner = ExperimentRunner()
    results = runner.run_ab(tasks=DEFAULT_TASKS, rounds=rounds)

    results_path = output_dir / "demo_results.json"
    report_path = output_dir / "demo_report.md"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(results, rounds), encoding="utf-8")

    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"\nDemo JSON written to {results_path}")
    print(f"Demo report written to {report_path}")


def _render_report(results: dict[str, Any], rounds: int) -> str:
    summary = results["summary"]
    text = summary["text"]
    structured = summary["structured"]
    environment = results.get("environment", {})
    delta = summary.get("structured_token_delta_pct", 0)
    advantage = "有优势" if delta < 0 else "暂未体现 token 优势"

    lines = [
        "# Multi-Agent Communication Demo Report",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Rounds: {rounds}",
        f"- OS: {environment.get('os_pretty_name', 'unknown')}",
        f"- Python: {environment.get('python_version', 'unknown')}",
        f"- LLM backend: {environment.get('llm_backend', 'required')}",
        f"- LLM model: {environment.get('llm_model', 'required')}",
        f"- Embedding backend: {environment.get('embedding_backend', 'hash')}",
        f"- Embedding model: {environment.get('embedding_model', 'hash')}",
        f"- Orchestrator: {environment.get('orchestrator', 'sequential')}",
        f"- Token count method: {text.get('token_count_method', 'char_approx_4')}",
        "",
        "## Demo Goal",
        "",
        "展示多 Agent 在纯文本通信与结构化通信下的协作流程，并比较消息数、通信 token、耗时、记忆命中、非文本状态传递等指标。",
        "",
        "## Requirement Coverage",
        "",
        "| Requirement | Current Status | Evidence |",
        "| --- | --- | --- |",
        "| 3+ Agent collaboration | Done | Planner, Researcher, Executor, Summarizer |",
        "| Text vs structured A/B | Done | Same tasks run in both modes |",
        "| Formal protocol | Done | message_id, task_id, parent_id, handshake, capability discovery |",
        "| LangGraph orchestration | Done when ORCHESTRATOR=langgraph | Metrics record orchestrator name |",
        "| Shared memory | Done | SQLite memory store |",
        "| Real embedding | Done when EMBEDDING_BACKEND=ollama | bge-m3 returns 1024-d vectors |",
        "| Non-text state transfer | Done | embedding/state refs/tool result counted |",
        "| CodeAct tools | Done | restricted Python tool execution |",
        "| Reproducible metrics | Prototype done | JSON and Markdown reports generated |",
        "",
        "## A/B Summary",
        "",
        "| Metric | Text | Structured |",
        "| --- | ---: | ---: |",
        f"| Runs | {text['runs']} | {structured['runs']} |",
        f"| Avg messages | {text['avg_messages']} | {structured['avg_messages']} |",
        f"| Avg chars | {text['avg_chars']} | {structured['avg_chars']} |",
        f"| Std chars | {text['std_chars']} | {structured['std_chars']} |",
        f"| Avg approx tokens | {text['avg_approx_tokens']} | {structured['avg_approx_tokens']} |",
        f"| Std approx tokens | {text['std_approx_tokens']} | {structured['std_approx_tokens']} |",
        f"| Min approx tokens | {text['min_approx_tokens']} | {structured['min_approx_tokens']} |",
        f"| Max approx tokens | {text['max_approx_tokens']} | {structured['max_approx_tokens']} |",
        f"| Avg elapsed ms | {text['avg_elapsed_ms']} | {structured['avg_elapsed_ms']} |",
        f"| Std elapsed ms | {text['std_elapsed_ms']} | {structured['std_elapsed_ms']} |",
        f"| Avg memory hits | {text['avg_memory_hits']} | {structured['avg_memory_hits']} |",
        f"| Avg non-text transfers | {text['avg_non_text_transfers']} | {structured['avg_non_text_transfers']} |",
        f"| Avg non-text transfer size | {text['avg_non_text_transfer_size']} | {structured['avg_non_text_transfer_size']} |",
        f"| Avg protocol events | {text['avg_protocol_events']} | {structured['avg_protocol_events']} |",
        f"| Avg protocol approx tokens | {text['avg_protocol_approx_tokens']} | {structured['avg_protocol_approx_tokens']} |",
        "",
        "## Main Finding",
        "",
        f"- Structured application-message token delta: **{delta}%**",
        f"- Current conclusion: **{advantage}**",
        "- Protocol negotiation overhead is reported separately from application messages.",
        "- Token counts use a transparent character-based approximation unless a real tokenizer is added.",
        "",
        "## Agent Trace Example",
        "",
    ]

    example = _first_structured_run(results)
    if example:
        lines.extend(_render_trace(example))

    lines.extend(
        [
            "",
            "## Remaining Work For Final Submission",
            "",
            "- Run this report with the deployed real LLM and embedding services on the target machine.",
            "- Replace approximate token counting with a tokenizer that matches the selected model.",
            "- Strengthen sandboxing if CodeAct runs untrusted code.",
            "- Implement true KV-cache or hidden-state reuse if the deployed model server exposes those states.",
            "- Record the exact target OS version; this run records it from /etc/os-release.",
        ]
    )
    return "\n".join(lines) + "\n"


def _first_structured_run(results: dict[str, Any]) -> dict[str, Any] | None:
    for run in results["runs"]:
        if run["mode"] == "structured":
            return run
    return None


def _render_trace(run: dict[str, Any]) -> list[str]:
    lines = [
        f"- Task: {run['task']}",
        "",
        "| From | To | Action | Refs | State Keys | Approx Tokens |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for message in run["metrics"].get("message_trace", []):
        state = message.get("state", {})
        state_keys = ", ".join(state.keys()) if isinstance(state, dict) else ""
        lines.append(
            "| {from_} | {to} | {action} | {refs} | {state_keys} | {tokens} |".format(
                from_=message.get("from"),
                to=message.get("to"),
                action=message.get("action"),
                refs=len(message.get("refs", [])),
                state_keys=state_keys,
                tokens=message.get("approx_tokens"),
            )
        )
    return lines


if __name__ == "__main__":
    main()
