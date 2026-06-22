from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multi_agent.llm_client import LLMClient
from multi_agent.memory import SharedMemory
from multi_agent.metrics import MetricsCollector
from multi_agent.protocol import Message, make_text_message
from multi_agent.state_exchange import StateStore
from multi_agent.tools import CodeActExecutor, ToolRegistry, build_codeact_for_task


@dataclass
class AgentContext:
    mode: str
    task_id: str
    memory: SharedMemory
    state_store: StateStore
    metrics: MetricsCollector
    enable_memory_search: bool = True
    enable_memory_write: bool = True
    enable_state_exchange: bool = True


class BaseAgent:
    def __init__(self, name: str, llm: LLMClient) -> None:
        self.name = name
        self.llm = llm

    def _ask(self, system: str, user: str) -> str:
        return self.llm.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )

    @property
    def capabilities(self) -> list[str]:
        return ["llm_chat"]


class PlannerAgent(BaseAgent):
    def plan(self, task: str, ctx: AgentContext) -> Message:
        prompt = f"Break this task into a concise research/execution plan:\n{task}"
        content = self._ask("You are a planning agent.", prompt)
        payload: dict[str, Any] = {
            "mt": "request",
            "a": "plan",
            "in": {"task": _short(task)},
            "out": _short(content),
            "refs": [],
        }
        return _message(ctx, self.name, "researcher", content, payload)

    @property
    def capabilities(self) -> list[str]:
        return ["planning", "task_decomposition", "llm_chat"]


class ResearchAgent(BaseAgent):
    def research(self, task: str, plan: Message, ctx: AgentContext) -> Message:
        hits = ctx.memory.search(task, limit=3) if ctx.enable_memory_search else []
        for hit in hits:
            ctx.metrics.record_memory_hit(hit.memory_id)

        memory_text = "\n".join(f"- {hit.summary}" for hit in hits) or "- no prior memory"
        prompt = (
            f"Task:\n{task}\n\nPlan:\n{plan.content}\n\n"
            f"Relevant memory:\n{memory_text}\n\n"
            "Return concise findings."
        )
        content = self._ask("You are a research agent.", prompt)
        memory_refs = [hit.memory_id for hit in hits]
        payload = {
            "mt": "request",
            "a": "research",
            "in": {"task": _short(task), "plan": _short(plan.content)},
            "out": _short(content),
            "refs": memory_refs,
        }

        if memory_refs:
            ctx.metrics.record_non_text_transfer("memory_refs", len(memory_refs))

        return _message(ctx, self.name, "executor", content, payload, parent_id=plan.message_id)

    @property
    def capabilities(self) -> list[str]:
        return ["memory_search", "analysis", "llm_chat"]


class ExecutorAgent(BaseAgent):
    def __init__(self, name: str, llm: LLMClient, codeact: CodeActExecutor | None = None) -> None:
        super().__init__(name, llm)
        self.codeact = codeact or CodeActExecutor()

    def execute(self, task: str, findings: Message, ctx: AgentContext) -> Message:
        prompt = (
            f"Task:\n{task}\n\nFindings:\n{findings.content}\n\n"
            "Produce a concrete answer and include reusable facts."
        )
        content = self._ask("You are an execution agent.", prompt)
        code = build_codeact_for_task(task, findings.content)
        tool_registry = ToolRegistry(memory=ctx.memory if ctx.enable_memory_search else None)
        tool_result = self.codeact.run(code, context=tool_registry.as_context())
        embedding = ctx.memory.embed(content)
        memory = None
        refs: list[str] = []
        state_payload: dict[str, Any] = {"embedding_dim": len(embedding), "tool_ok": tool_result.ok}

        if ctx.enable_memory_write:
            memory = ctx.memory.add(
                source_agent=self.name,
                task_topic=task,
                summary=content,
                tags=_tags_for_task(task),
                embedding=embedding,
            )
            refs.append(memory.memory_id)

        if ctx.enable_state_exchange:
            embedding_state = ctx.state_store.put(
                producer_agent=self.name,
                task_id=ctx.task_id,
                state_type="embedding",
                payload=embedding,
                metadata={
                    "memory_id": memory.memory_id if memory else None,
                    "dim": len(embedding),
                    "usage": "semantic_retrieval",
                },
            )
            tool_state = ctx.state_store.put(
                producer_agent=self.name,
                task_id=ctx.task_id,
                state_type="codeact_result",
                payload=tool_result.to_dict(),
                metadata={
                    "tool": "codeact/python",
                    "ok": tool_result.ok,
                    "usage": "execution_evidence",
                },
            )
            ctx.metrics.record_non_text_transfer("embedding", embedding_state.size_bytes, embedding_state.state_id)
            ctx.metrics.record_non_text_transfer("codeact_result", tool_state.size_bytes, tool_state.state_id)
            refs.extend([embedding_state.state_id, tool_state.state_id])
            state_payload.update(
                {
                    "embedding_ref": embedding_state.state_id,
                    "tool_result_ref": tool_state.state_id,
                    "sizes": {
                        "embedding_bytes": embedding_state.size_bytes,
                        "tool_result_bytes": tool_state.size_bytes,
                    },
                }
            )
        else:
            state_payload["state_exchange"] = "disabled"

        payload = {
            "mt": "response",
            "a": "execute",
            "in": {"task": _short(task), "findings": _short(findings.content)},
            "out": _short(content),
            "refs": refs,
            "state": state_payload,
        }
        return _message(ctx, self.name, "summarizer", content, payload, parent_id=findings.message_id)

    @property
    def capabilities(self) -> list[str]:
        return [
            "codeact_python",
            "read_file",
            "search_files",
            "load_json",
            "load_csv",
            "make_markdown_table",
            "compute_numeric_metrics",
            "search_memory",
            "memory_write",
            "embedding_state",
            "llm_chat",
        ]


class SummarizerAgent(BaseAgent):
    def summarize(self, task: str, execution: Message, ctx: AgentContext) -> Message:
        prompt = (
            f"Task:\n{task}\n\nExecution result:\n{execution.content}\n\n"
            "Summarize the final answer in 3 bullet points."
        )
        content = self._ask("You are a summarization agent.", prompt)
        payload = {
            "mt": "response",
            "a": "summarize",
            "in": {"task": _short(task), "execution": _short(execution.content)},
            "out": _short(content),
            "refs": execution.payload.get("refs", []),
        }
        return _message(ctx, self.name, "user", content, payload, parent_id=execution.message_id)

    @property
    def capabilities(self) -> list[str]:
        return ["summarization", "llm_chat"]


def _message(
    ctx: AgentContext,
    sender: str,
    receiver: str,
    content: str,
    payload: dict[str, Any],
    parent_id: str | None = None,
) -> Message:
    if ctx.mode == "structured":
        return Message.structured(
            sender=sender,
            receiver=receiver,
            payload=payload,
            task_id=ctx.task_id,
            parent_id=parent_id,
        )
    return make_text_message(
        sender=sender,
        receiver=receiver,
        content=_text_handoff(sender, receiver, content, payload),
        task_id=ctx.task_id,
        parent_id=parent_id,
    )


def _tags_for_task(task: str) -> list[str]:
    tags = []
    lowered = task.lower()
    for keyword in ("agent", "memory", "protocol", "evaluation", "sandbox", "llm"):
        if keyword in lowered:
            tags.append(keyword)
    return tags or ["general"]


def _short(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[: limit - 3]}..."


def _text_handoff(sender: str, receiver: str, content: str, payload: dict[str, Any]) -> str:
    """Natural-language handoff used as the baseline text collaboration mode."""

    lines = [
        f"{sender} is handing off work to {receiver}.",
        f"The action being performed is {payload.get('a', 'unknown')}.",
    ]
    if payload.get("in"):
        lines.append(f"The relevant input context is: {payload['in']}.")
    lines.append(f"The result is: {content}")
    if payload.get("refs"):
        lines.append(
            "The following memory or state references may be useful, but they are described here in text form: "
            f"{payload['refs']}."
        )
    if payload.get("state"):
        lines.append(f"The intermediate state summary is: {payload['state']}.")
    lines.append("Please read the full handoff text above before continuing the multi-agent task.")
    return "\n".join(lines)
