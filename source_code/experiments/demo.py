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
    report_path = output_dir / "demo_report.txt"
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
    memory_stats = _memory_stats(results.get("memory", []))
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
        f"- Memory path: {environment.get('memory_path', 'data/memory.sqlite')}",
        f"- State path: {environment.get('state_path', 'data/state.sqlite')}",
        f"- State backend: {environment.get('state_backend', 'shared_memory')}",
        f"- CodeAct sandbox: {environment.get('codeact_sandbox', 'subprocess')}",
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
        "| Formal protocol | Done | message_id, task_id, parent_id, message type, handshake, capability discovery, error code table |",
        "| LangGraph orchestration | Done when ORCHESTRATOR=langgraph | Metrics record orchestrator name |",
        "| Shared memory | Done | SQLite memory store |",
        "| Real embedding | Done when EMBEDDING_BACKEND=ollama | bge-m3 returns 1024-d vectors |",
        "| Non-text state transfer | Done | StateStore saves embedding/tool result and protocol carries state refs |",
        "| CodeAct tools | Done | restricted Python tool execution |",
        "| Reproducible metrics | Done | JSON and Markdown reports generated with environment metadata |",
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
        "- Token counts use the configured method shown above; set TOKEN_COUNT_METHOD=tiktoken when tiktoken is installed.",
        f"- State records persisted: **{len(results.get('states', []))}**",
        f"- Shared memories persisted: **{len(results.get('memory', []))}**",
        f"- Avg memory links: **{memory_stats['avg_links']}**, total access count: **{memory_stats['total_access_count']}**",
        "",
        "## Agent Trace Example",
        "",
    ]

    example = _first_structured_run(results)
    if example:
        lines.extend(_render_trace(example))
        lines.extend(_render_state_refs(example, results))

    lines.extend(
        [
            "",
            "## Final Submission Checklist",
            "",
            "- Run at least 10 rounds on openEuler 24.03-LTS-SP3 and keep the generated JSON/Markdown artifacts.",
            "- Use EMBEDDING_BACKEND=ollama and EMBEDDING_TIMEOUT=60 or higher for the formal embedding experiment.",
            "- Use TOKEN_COUNT_METHOD=tiktoken if the selected model tokenizer is available; otherwise report character and heuristic token counts together.",
            "- Include docs/system_design.md, docs/deployment.md, docs/experiment_report.md, reports/demo_report.md, and the demo video.",
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


def _render_state_refs(run: dict[str, Any], results: dict[str, Any]) -> list[str]:
    states_by_id = {state.get("state_id"): state for state in results.get("states", [])}
    rows = []
    for state_id in run.get("states", []):
        state = states_by_id.get(state_id)
        if state:
            rows.append(state)
    if not rows:
        return []
    lines = [
        "",
        "## Non-Text State Exchange Example",
        "",
        "| State ID | Type | Producer | Size Bytes | Metadata |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {state_id} | {state_type} | {producer} | {size} | {metadata} |".format(
                state_id=row.get("state_id"),
                state_type=row.get("state_type"),
                producer=row.get("producer_agent"),
                size=row.get("size_bytes"),
                metadata=json.dumps(row.get("metadata", {}), ensure_ascii=False),
            )
        )
    return lines


def _memory_stats(memories: list[dict[str, Any]]) -> dict[str, Any]:
    if not memories:
        return {"avg_links": 0.0, "total_access_count": 0}
    avg_links = sum(len(memory.get("links", [])) for memory in memories) / len(memories)
    total_access = sum(int(memory.get("access_count", 0) or 0) for memory in memories)
    return {"avg_links": round(avg_links, 3), "total_access_count": total_access}


if __name__ == "__main__":
    main()
