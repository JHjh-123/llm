from __future__ import annotations

import os
import uuid
import json
from dataclasses import asdict
from statistics import mean, pstdev
from typing import Any

import grpc
from multi_agent.proto import agent_pb2, agent_pb2_grpc

from multi_agent.agents import AgentContext, ExecutorAgent, PlannerAgent, ResearchAgent, SummarizerAgent, VerifierAgent, RouterAgent, SecurityReviewerAgent, DebuggerAgent, MemoryArchivistAgent
from multi_agent.environment import collect_environment
from multi_agent.llm_client import build_llm_from_env
from multi_agent.metrics import TOKEN_COUNT_METHOD
from multi_agent.memory import SharedMemory
from multi_agent.metrics import MetricsCollector
from multi_agent.orchestrator import AgentBundle, build_orchestrator
from multi_agent.protocol import make_handshake, make_protocol_mapping, Message
from multi_agent.state_exchange import StateStore


class RemoteAgentWrapper:
    """gRPC Client wrapper that forwards agent requests to AgentService.

    Matches the exact interface of the local agent objects.
    """

    def __init__(self, name: str, service_url: str) -> None:
        self.name = name
        # Clean service URL to remove protocol schemes
        if service_url.startswith("http://"):
            service_url = service_url[len("http://"):]
        elif service_url.startswith("https://"):
            service_url = service_url[len("https://"):]
        self.service_url = service_url
        self._channel = None
        self._stub = None

    def _get_stub(self):
        if self._channel is None:
            self._channel = grpc.insecure_channel(self.service_url)
            self._stub = agent_pb2_grpc.AgentServiceStub(self._channel)
        return self._stub

    def route(self, task: str, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.RouteRequest(task=task, mode=ctx.mode, task_id=ctx.task_id)
        try:
            pb_msg = stub.RouteTask(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC RouteTask failed: {e}") from e

    def plan(self, task: str, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.PlanRequest(task=task, mode=ctx.mode, task_id=ctx.task_id)
        try:
            pb_msg = stub.PlanTask(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC PlanTask failed: {e}") from e

    def research(self, task: str, plan: Message, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.ResearchRequest(
            task=task,
            plan=self._message_to_protobuf(plan),
            mode=ctx.mode,
            task_id=ctx.task_id
        )
        try:
            pb_msg = stub.ResearchTask(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC ResearchTask failed: {e}") from e

    def execute(self, task: str, findings: Message, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.ExecuteRequest(
            task=task,
            findings=self._message_to_protobuf(findings),
            mode=ctx.mode,
            task_id=ctx.task_id
        )
        try:
            pb_msg = stub.ExecuteTask(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC ExecuteTask failed: {e}") from e

    def summarize(self, task: str, execution: Message, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.SummarizeRequest(
            task=task,
            execution=self._message_to_protobuf(execution),
            mode=ctx.mode,
            task_id=ctx.task_id
        )
        try:
            pb_msg = stub.SummarizeTask(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC SummarizeTask failed: {e}") from e

    def verify(self, task: str, plan: Message, summary: Message, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.VerifyRequest(
            task=task,
            plan=self._message_to_protobuf(plan),
            summary=self._message_to_protobuf(summary),
            mode=ctx.mode,
            task_id=ctx.task_id
        )
        try:
            pb_msg = stub.VerifyTask(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC VerifyTask failed: {e}") from e

    def review(self, task: str, code: str, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.ReviewRequest(task=task, code=code, mode=ctx.mode, task_id=ctx.task_id)
        try:
            pb_msg = stub.ReviewCode(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC ReviewCode failed: {e}") from e

    def debug(self, task: str, code: str, error: str, stdout: str, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.DebugRequest(
            task=task,
            code=code,
            error=error,
            stdout=stdout,
            mode=ctx.mode,
            task_id=ctx.task_id
        )
        try:
            pb_msg = stub.DebugCode(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC DebugCode failed: {e}") from e

    def archive(self, ctx: AgentContext) -> Message:
        stub = self._get_stub()
        req = agent_pb2.ArchiveRequest(mode=ctx.mode, task_id=ctx.task_id)
        try:
            pb_msg = stub.ArchiveMemory(req, timeout=600)
            return self._protobuf_to_message(pb_msg)
        except Exception as e:
            raise RuntimeError(f"gRPC ArchiveMemory failed: {e}") from e

    @property
    def capabilities(self) -> list[str]:
        return ["remote_execution"]

    def _message_to_protobuf(self, msg: Message) -> agent_pb2.Message:
        return agent_pb2.Message(
            message_id=msg.message_id,
            task_id=msg.task_id,
            parent_id=msg.parent_id or "",
            sender=msg.sender,
            receiver=msg.receiver,
            mode=msg.mode,
            content=msg.content,
            payload_json=json.dumps(msg.payload, ensure_ascii=False) if msg.payload else "{}",
            created_at=msg.created_at,
            error=json.dumps(msg.error, ensure_ascii=False) if msg.error else ""
        )

    def _protobuf_to_message(self, pb_msg: agent_pb2.Message) -> Message:
        payload = {}
        if pb_msg.payload_json:
            try:
                payload = json.loads(pb_msg.payload_json)
            except Exception:
                pass
                
        error = None
        if pb_msg.error:
            try:
                error = json.loads(pb_msg.error)
            except Exception:
                error = {"message": pb_msg.error}
                
        return Message(
            message_id=pb_msg.message_id,
            task_id=pb_msg.task_id,
            parent_id=pb_msg.parent_id if pb_msg.parent_id else None,
            sender=pb_msg.sender,
            receiver=pb_msg.receiver,
            mode=pb_msg.mode,
            content=pb_msg.content,
            payload=payload,
            created_at=pb_msg.created_at,
            error=error
        )


class ExperimentRunner:
    def __init__(self) -> None:
        llm = build_llm_from_env()
        self.memory = SharedMemory(reset=_env_bool("MEMORY_RESET", default=False))
        self.state_store = StateStore(reset=_env_bool("STATE_RESET", default=_env_bool("MEMORY_RESET", default=False)))
        
        self.deployment = os.getenv("AGENT_DEPLOYMENT", "local").lower()
        self.service_url = os.getenv("AGENT_SERVICE_URL", "localhost:50051")


        if self.deployment == "microservice":
            print(f"Initializing ExperimentRunner in distributed MICROSERVICE mode connecting to {self.service_url}")
            self.planner = RemoteAgentWrapper("planner", self.service_url)
            self.researcher = RemoteAgentWrapper("researcher", self.service_url)
            self.executor = RemoteAgentWrapper("executor", self.service_url)
            self.summarizer = RemoteAgentWrapper("summarizer", self.service_url)
            self.verifier = RemoteAgentWrapper("verifier", self.service_url)
            self.router = RemoteAgentWrapper("router", self.service_url)
            self.security_reviewer = RemoteAgentWrapper("security_reviewer", self.service_url)
            self.debugger = RemoteAgentWrapper("debugger", self.service_url)
            self.archivist = RemoteAgentWrapper("archivist", self.service_url)
        else:
            self.planner = PlannerAgent("planner", llm)
            self.researcher = ResearchAgent("researcher", llm)
            self.executor = ExecutorAgent("executor", llm)
            self.summarizer = SummarizerAgent("summarizer", llm)
            self.verifier = VerifierAgent("verifier", llm)
            self.router = RouterAgent("router", llm)
            self.security_reviewer = SecurityReviewerAgent("security_reviewer", llm)
            self.debugger = DebuggerAgent("debugger", llm)
            self.archivist = MemoryArchivistAgent("archivist", llm)

        self.agents: AgentBundle = {
            "planner": self.planner,
            "researcher": self.researcher,
            "executor": self.executor,
            "summarizer": self.summarizer,
            "verifier": self.verifier,
            "router": self.router,
            "security_reviewer": self.security_reviewer,
            "debugger": self.debugger,
            "archivist": self.archivist,
        }
        self.orchestrator = build_orchestrator(os.getenv("ORCHESTRATOR", "sequential").lower(), self.agents)
        self._protocol_negotiated = False

    def run_task(self, task: str, mode: str) -> dict[str, Any]:
        metrics = MetricsCollector(mode=mode, task=task)
        metrics.set_orchestrator(self.orchestrator.name)
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        ctx = AgentContext(
            mode=mode,
            task_id=task_id,
            memory=self.memory,
            state_store=self.state_store,
            metrics=metrics,
            enable_memory_search=_env_bool("FEATURE_MEMORY_SEARCH", default=True),
            enable_memory_write=_env_bool("FEATURE_MEMORY_WRITE", default=True),
            enable_state_exchange=_env_bool("FEATURE_STATE_EXCHANGE", default=True),
            security_reviewer=self.security_reviewer,
            debugger=self.debugger,
        )

        protocol_messages = []
        if mode == "structured" and self._should_negotiate_protocol():
            protocol_messages = self._negotiate_protocol(task_id)
            for message in protocol_messages:
                metrics.record_protocol_event(message)
            self._protocol_negotiated = True

        summary = self.orchestrator.run(task, ctx, metrics)
        if ctx.enable_memory_write:
            try:
                archive_msg = self.archivist.archive(ctx)
                metrics.record_message(archive_msg)
            except Exception:
                pass
        return {
            "task": task,
            "task_id": task_id,
            "mode": mode,
            "final": summary.content,
            "metrics": metrics.finish(),
            "protocol": [message.to_wire() for message in protocol_messages],
            "states": [state.state_id for state in self.state_store.list_for_task(task_id)],
        }

    def run_ab(self, tasks: list[str], rounds: int = 10) -> dict[str, Any]:
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
            "states": self.state_store.to_dict(),
            "environment": collect_environment(),
        }

    def _negotiate_protocol(self, task_id: str) -> list[Any]:
        messages = [
            make_protocol_mapping(task_id),
        ]
        for agent in self.agents.values():
            # Get capabilities safely
            caps = agent.capabilities if hasattr(agent, "capabilities") else ["remote_execution"]
            messages.append(make_handshake(task_id, agent.name, caps))
        return messages

    def _should_negotiate_protocol(self) -> bool:
        if not _env_bool("PROTOCOL_SESSION_CACHE", default=True):
            return True
        return not self._protocol_negotiated


def _summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {"text": [], "structured": []}
    for run in runs:
        by_mode[run["mode"]].append(run["metrics"])

    summary = {}
    for mode, metrics in by_mode.items():
        summary[mode] = {
            "runs": len(metrics),
            "token_count_method": TOKEN_COUNT_METHOD,
            "avg_messages": _avg(metrics, "message_count"),
            "std_messages": _std(metrics, "message_count"),
            "avg_chars": _avg(metrics, "char_count"),
            "std_chars": _std(metrics, "char_count"),
            "avg_approx_tokens": _avg(metrics, "approx_token_count"),
            "std_approx_tokens": _std(metrics, "approx_token_count"),
            "min_approx_tokens": _min(metrics, "approx_token_count"),
            "max_approx_tokens": _max(metrics, "approx_token_count"),
            "avg_elapsed_ms": _avg(metrics, "elapsed_ms"),
            "std_elapsed_ms": _std(metrics, "elapsed_ms"),
            "min_elapsed_ms": _min(metrics, "elapsed_ms"),
            "max_elapsed_ms": _max(metrics, "elapsed_ms"),
            "avg_memory_hits": _avg(metrics, "memory_hit_count"),
            "std_memory_hits": _std(metrics, "memory_hit_count"),
            "avg_memory_graph_hits": _avg(metrics, "memory_graph_hit_count"),
            "std_memory_graph_hits": _std(metrics, "memory_graph_hit_count"),
            "avg_non_text_transfers": _avg(metrics, "non_text_transfer_count"),
            "std_non_text_transfers": _std(metrics, "non_text_transfer_count"),
            "avg_non_text_transfer_size": _avg(metrics, "non_text_transfer_size"),
            "avg_protocol_events": _avg(metrics, "protocol_event_count"),
            "avg_protocol_chars": _avg(metrics, "protocol_char_count"),
            "avg_protocol_approx_tokens": _avg(metrics, "protocol_approx_token_count"),
            "avg_dynamic_codeact": _avg(metrics, "dynamic_codeact_count"),
            "avg_fallback_codeact": _avg(metrics, "fallback_codeact_count"),
        }

    text_tokens = summary["text"]["avg_approx_tokens"]
    structured_tokens = summary["structured"]["avg_approx_tokens"]
    if text_tokens:
        summary["structured_token_delta_pct"] = round((structured_tokens - text_tokens) / text_tokens * 100, 2)
    return summary


def _avg(metrics: list[dict[str, Any]], key: str) -> float:
    return round(mean(float(item[key]) for item in metrics), 3) if metrics else 0.0


def _std(metrics: list[dict[str, Any]], key: str) -> float:
    return round(pstdev(float(item[key]) for item in metrics), 3) if len(metrics) > 1 else 0.0


def _min(metrics: list[dict[str, Any]], key: str) -> float:
    return round(min(float(item[key]) for item in metrics), 3) if metrics else 0.0


def _max(metrics: list[dict[str, Any]], key: str) -> float:
    return round(max(float(item[key]) for item in metrics), 3) if metrics else 0.0


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
