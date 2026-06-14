from __future__ import annotations

import os
import uuid
from statistics import mean
from typing import Any

from multi_agent.agents import AgentContext, ExecutorAgent, PlannerAgent, ResearchAgent, SummarizerAgent
from multi_agent.llm_client import build_llm_from_env
from multi_agent.memory import SharedMemory
from multi_agent.metrics import MetricsCollector
from multi_agent.orchestrator import AgentBundle, build_orchestrator
from multi_agent.protocol import make_handshake, make_protocol_mapping


class ExperimentRunner:
    def __init__(self) -> None:
        llm = build_llm_from_env()
        self.memory = SharedMemory(reset=_env_bool("MEMORY_RESET", default=False))
        self.planner = PlannerAgent("planner", llm)
        self.researcher = ResearchAgent("researcher", llm)
        self.executor = ExecutorAgent("executor", llm)
        self.summarizer = SummarizerAgent("summarizer", llm)
        self.agents: AgentBundle = {
            "planner": self.planner,
            "researcher": self.researcher,
            "executor": self.executor,
            "summarizer": self.summarizer,
        }
        self.orchestrator = build_orchestrator(os.getenv("ORCHESTRATOR", "sequential").lower(), self.agents)

    def run_task(self, task: str, mode: str) -> dict[str, Any]:
        metrics = MetricsCollector(mode=mode, task=task)
        metrics.set_orchestrator(self.orchestrator.name)
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        ctx = AgentContext(mode=mode, task_id=task_id, memory=self.memory, metrics=metrics)

        protocol_messages = []
        if mode == "structured":
            protocol_messages = self._negotiate_protocol(task_id)
            for message in protocol_messages:
                metrics.record_protocol_event(message)

        summary = self.orchestrator.run(task, ctx, metrics)
        return {
            "task": task,
            "task_id": task_id,
            "mode": mode,
            "final": summary.content,
            "metrics": metrics.finish(),
            "protocol": [message.to_wire() for message in protocol_messages],
        }

    def run_ab(self, tasks: list[str], rounds: int = 3) -> dict[str, Any]:
        runs = []
        for round_index in range(rounds):
            for task in tasks:
                for mode in ("text", "structured"):
                    run = self.run_task(task=task, mode=mode)
                    run["round"] = round_index + 1
                    runs.append(run)

        return {
            "summary": _summarize(runs),
            "runs": runs,
            "memory": self.memory.to_dict(),
        }

    def _negotiate_protocol(self, task_id: str) -> list[Any]:
        messages = [
            make_protocol_mapping(task_id),
        ]
        for agent in self.agents.values():
            messages.append(make_handshake(task_id, agent.name, agent.capabilities))
        return messages


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {"text": [], "structured": []}
    for run in runs:
        by_mode[run["mode"]].append(run["metrics"])

    summary = {}
    for mode, metrics in by_mode.items():
        summary[mode] = {
            "runs": len(metrics),
            "avg_messages": _avg(metrics, "message_count"),
            "avg_chars": _avg(metrics, "char_count"),
            "avg_approx_tokens": _avg(metrics, "approx_token_count"),
            "avg_elapsed_ms": _avg(metrics, "elapsed_ms"),
            "avg_memory_hits": _avg(metrics, "memory_hit_count"),
            "avg_non_text_transfers": _avg(metrics, "non_text_transfer_count"),
            "avg_protocol_events": _avg(metrics, "protocol_event_count"),
            "avg_protocol_chars": _avg(metrics, "protocol_char_count"),
            "avg_protocol_approx_tokens": _avg(metrics, "protocol_approx_token_count"),
        }

    text_tokens = summary["text"]["avg_approx_tokens"]
    structured_tokens = summary["structured"]["avg_approx_tokens"]
    if text_tokens:
        summary["structured_token_delta_pct"] = round((structured_tokens - text_tokens) / text_tokens * 100, 2)
    return summary


def _avg(metrics: list[dict[str, Any]], key: str) -> float:
    return round(mean(float(item[key]) for item in metrics), 3) if metrics else 0.0


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
