from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from multi_agent.llm_client import LLMClient
from multi_agent.memory import SharedMemory
from multi_agent.metrics import MetricsCollector
from multi_agent.protocol import Message, make_text_message
from multi_agent.state_exchange import StateStore
from multi_agent.tools import CodeActExecutor, ToolRegistry, build_codeact_for_task, build_codeact_from_plan, extract_file_paths
import ast
import os
import re
import json

def _extract_code(text: str) -> str | None:
    # Match code blocks of the form ```python ... ```
    match = re.search(r"```python\s+(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback to general block
    match = re.search(r"```\s+(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback to raw lines if there's python looking code
    if "import " in text or "def " in text or "print(" in text:
        return text.strip()
    return None


def _code_from_tool_plan(task: str, findings: str, text: str) -> str | None:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    candidate = match.group(0) if match else candidate
    try:
        plan = json.loads(candidate)
    except Exception:
        try:
            pythonish = (
                candidate.replace("true", "True")
                .replace("false", "False")
                .replace("null", "None")
            )
            plan = ast.literal_eval(pythonish)
        except Exception:
            lowered = text.lower()
            if not any(token in lowered for token in ("markdown", "memory", "metric", "table", "file")):
                return None
            plan = {
                "read_markdown_files": "markdown" in lowered or "file" in lowered,
                "search_memory": "memory" in lowered,
                "compute_metrics": "metric" in lowered or "compute" in lowered,
                "make_table": "table" in lowered,
            }
    if not isinstance(plan, dict):
        return None
    allowed = {"read_markdown_files", "search_memory", "compute_metrics", "make_table"}
    normalized = {key: bool(plan.get(key, True)) for key in allowed}
    return build_codeact_from_plan(task, findings, normalized)


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
    security_reviewer: Any = None
    debugger: Any = None


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
        query_embedding = None
        if ctx.enable_state_exchange:
            if plan.payload:
                for ref in plan.payload.get("refs", []):
                    if isinstance(ref, str) and ref.startswith("state_"):
                        state_rec = ctx.state_store.get(ref)
                        if state_rec and state_rec.state_type == "embedding":
                            query_embedding = ctx.state_store.read_shared_vector(state_rec)
                            if query_embedding:
                                ctx.metrics.record_non_text_transfer(
                                    "state_read_embedding",
                                    state_rec.size_bytes,
                                    state_rec.state_id
                                )
                                break
            
        default_memory_limit = "2" if ctx.mode == "structured" else "1"
        memory_limit = max(1, int(os.getenv("MEMORY_SEARCH_LIMIT_STRUCTURED" if ctx.mode == "structured" else "MEMORY_SEARCH_LIMIT", default_memory_limit)))
        if query_embedding is not None and ctx.enable_memory_search:
            hits = ctx.memory.search(query_embedding, limit=memory_limit)
        else:
            hits = ctx.memory.search(task, limit=memory_limit) if ctx.enable_memory_search else []
        for hit in hits:
            ctx.metrics.record_memory_hit(hit.memory_id)
            if hit.keywords or hit.links:
                ctx.metrics.record_memory_graph_hit(hit.memory_id)

        memory_refs = [hit.memory_id for hit in hits]
        memory_text = "\n".join(f"- {hit.memory_id}: {_short(hit.summary, 180)}" for hit in hits) or "- no prior memory"
        prompt = (
            f"Task:\n{task}\n\nPlan:\n{plan.content}\n\n"
            f"Relevant shared memory:\n{memory_text}\n\n"
            "Return concise findings. If memory is relevant, explicitly reuse it and avoid repeating details that can be represented by memory IDs."
        )
        content = self._ask("You are a research agent.", prompt)
        payload_out = _short(content)
        state: dict[str, Any] = {}
        if ctx.mode == "structured" and memory_refs:
            payload_out = f"mem:{len(memory_refs)} {_short(content, 72)}"
            state["mem_hits"] = len(memory_refs)
            graph_refs = [hit.memory_id for hit in hits if hit.keywords or hit.links]
            if graph_refs:
                state["graph_refs"] = graph_refs[:3]
        payload = {
            "mt": "request",
            "a": "research",
            "in": {"task": _short(task), "plan": _short(plan.content)},
            "out": payload_out,
            "refs": memory_refs,
        }
        if state:
            payload["state"] = state

        if ctx.mode == "structured" and memory_refs:
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
        findings_context = _agent_context(findings, limit=420 if ctx.mode == "structured" else 1200)
        answer_instruction = (
            "Produce a concrete answer in at most 2 concise bullets. Use refs/state IDs as evidence handles instead of restating prior details."
            if ctx.mode == "structured"
            else "Produce a concrete answer and include reusable facts. If findings include memory IDs, use them as evidence references instead of restating long prior reasoning."
        )
        prompt = f"Task:\n{task}\n\nFindings:\n{findings_context}\n\n{answer_instruction}"
        content = self._ask("You are an execution agent.", prompt)
        reused_memory_refs = _memory_refs(findings)

        # 1. Ask the LLM to select a tool plan, then compile it to safe Python.
        code_prompt = (
            f"Task:\n{task}\n\n"
            f"Findings:\n{findings_context}\n\n"
            "Choose which safe tools should be used to verify the findings or compute compact evidence. "
            f"Existing memory references available: {len(reused_memory_refs)}. "
            "If existing memory references are available, set search_memory to false unless another memory lookup is truly necessary. "
            "Return ONLY a JSON object with boolean fields:\n"
            "{\n"
            '  "read_markdown_files": true or false,\n'
            '  "search_memory": true or false,\n'
            '  "compute_metrics": true or false,\n'
            '  "make_table": true or false\n'
            "}\n"
            "Do not return Python code or explanatory prose."
        )

        tool_registry = ToolRegistry(memory=ctx.memory if ctx.enable_memory_search else None)
        code = None
        tool_result = None
        code_from_dynamic_plan = False
        code_source = "dynamic_llm"

        if ctx.mode == "structured":
            code = build_codeact_from_plan(task, findings_context, _protocol_tool_plan(task, reused_memory_refs))
            tool_result = self.codeact.run(code, context=tool_registry.as_context())
            code_from_dynamic_plan = True
            code_source = "protocol_plan"
        else:
            current_prompt = code_prompt

            # Reflection / retry loop with Security Reviewer and Debugger
            for attempt in range(3):
                llm_code_response = self._ask("You are an execution agent selecting safe tools.", current_prompt)
                extracted_code = _code_from_tool_plan(task, findings_context, llm_code_response)
                code_from_dynamic_plan = extracted_code is not None
                if extracted_code is None:
                    extracted_code = _extract_code(llm_code_response)
                    code_from_dynamic_plan = False

                if not extracted_code:
                    current_prompt = code_prompt + "\n\nError: No valid JSON tool plan found. Return only the JSON object."
                    continue

                # 1. Security Review Check
                security_reviewer = getattr(ctx, "security_reviewer", None)
                if security_reviewer:
                    review_msg = security_reviewer.review(task, extracted_code, ctx)
                    ctx.metrics.record_message(review_msg)
                    approved = True
                    feedback = ""
                    if review_msg.payload and "state" in review_msg.payload:
                        approved = bool(review_msg.payload["state"].get("approved", True))
                        feedback = str(review_msg.payload["state"].get("feedback", ""))

                    if not approved:
                        current_prompt = (
                            f"Task:\n{task}\n\n"
                            f"Your Python code failed security review.\n"
                            f"Code attempted:\n```python\n{extracted_code}\n```\n\n"
                            f"Safety Feedback: {feedback}\n\n"
                            "Please correct the security violations and output a safe Python script inside a ```python and ``` block."
                        )
                        continue

                # 2. Subprocess Sandbox Execution
                run_result = self.codeact.run(extracted_code, context=tool_registry.as_context())
                if run_result.ok:
                    code = extracted_code
                    tool_result = run_result
                    break

                # 3. Debugger Reflection
                debugger = getattr(ctx, "debugger", None)
                if debugger:
                    debug_msg = debugger.debug(task, extracted_code, run_result.error or "Unknown error", run_result.stdout or "", ctx)
                    ctx.metrics.record_message(debug_msg)
                    explanation = ""
                    correction = ""
                    if debug_msg.payload and "state" in debug_msg.payload:
                        explanation = str(debug_msg.payload["state"].get("explanation", ""))
                        correction = str(debug_msg.payload["state"].get("correction", ""))

                    current_prompt = (
                        f"Task:\n{task}\n\n"
                        f"Your Python code failed to run.\n"
                        f"Code attempted:\n```python\n{extracted_code}\n```\n\n"
                        f"Debugger Explanation: {explanation}\n"
                        f"Debugger Suggested Correction:\n{correction}\n\n"
                        "Please rewrite the Python code to correct the error and output it inside a ```python and ``` block."
                    )
                else:
                    current_prompt = (
                        f"Task:\n{task}\n\n"
                        f"Your Python code failed to run.\n"
                        f"Code attempted:\n```python\n{extracted_code}\n```\n\n"
                        f"Error encountered: {run_result.error}\n"
                        f"Stdout captured: {run_result.stdout}\n\n"
                        "Please analyze the failure, correct the code, and output a new corrected Python script inside a ```python and ``` block."
                    )

        # Failsafe fallback keeps the task running, but metrics report when dynamic CodeAct did not succeed.
        is_dynamic = tool_result is not None and tool_result.ok and code_from_dynamic_plan
        if not is_dynamic:
            code = build_codeact_for_task(task, findings_context)
            tool_result = self.codeact.run(code, context=tool_registry.as_context())
            ctx.metrics.record_fallback_codeact()
            code_source = "fallback_template"
        else:
            ctx.metrics.record_dynamic_codeact()
            if code_source != "protocol_plan":
                code_source = "dynamic_llm"

        tool_evidence = _tool_evidence_text(tool_result.to_dict())
        if tool_evidence:
            content = f"{content}\n\nTool evidence:\n{tool_evidence}"

        embedding_source_text = _embedding_source_text(task, content)
        embedding = ctx.memory.embed(embedding_source_text)
        memory = None
        refs: list[str] = []
        state_payload: dict[str, Any] = {"ok": tool_result.ok}
        if ctx.mode == "structured" and tool_evidence:
            state_payload["evidence"] = _short(tool_evidence, 220)

        existing_topic_memory = None
        write_policy = os.getenv("MEMORY_WRITE_POLICY", "topic_once").strip().lower()
        if ctx.enable_memory_write and write_policy == "topic_once":
            existing_topic_memory = ctx.memory.latest_for_topic(self.name, task)
            should_write_memory = existing_topic_memory is None
        elif ctx.enable_memory_write and write_policy == "on_miss":
            should_write_memory = not reused_memory_refs
        else:
            should_write_memory = ctx.enable_memory_write and (
                not reused_memory_refs or _env_bool("MEMORY_WRITE_ON_REUSE", default=True)
            )
        if should_write_memory:
            memory = ctx.memory.add(
                source_agent=self.name,
                task_topic=task,
                summary=content,
                tags=_tags_for_task(task),
                embedding=embedding,
            )
            refs.append(memory.memory_id)
        elif reused_memory_refs:
            refs.extend(reused_memory_refs[:3])
            state_payload["reused_memory_refs"] = reused_memory_refs[:3]
        elif existing_topic_memory:
            refs.append(existing_topic_memory.memory_id)
            state_payload["reused_memory_refs"] = [existing_topic_memory.memory_id]

        if ctx.enable_state_exchange:
            tool_state = ctx.state_store.put(
                producer_agent=self.name,
                task_id=ctx.task_id,
                state_type="codeact_result",
                payload=_compact_tool_result(tool_result.to_dict(), ctx.mode),
                metadata={
                    "tool": "codeact/python",
                    "ok": tool_result.ok,
                    "usage": "execution_evidence",
                    "code_source": code_source,
                },
            )
            ctx.metrics.record_non_text_transfer("codeact_result", tool_state.size_bytes, tool_state.state_id)
            refs.append(tool_state.state_id)
            state_payload.update(
                {
                    "tool": tool_state.state_id,
                    "bytes": [tool_state.size_bytes],
                }
            )
        else:
            state_payload["sx"] = "off"

        output = _short(content)
        if ctx.mode == "structured" and ctx.enable_state_exchange:
            output = f"state:{len(refs)} {_short(tool_evidence or content, 110)}"
        payload = {
            "mt": "response",
            "a": "execute",
            "in": {"task": _short(task), "findings": _short(findings_context)},
            "out": output,
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
        execution_context = _agent_context(execution, limit=420 if ctx.mode == "structured" else 1200)
        prompt = (
            f"Task:\n{task}\n\nExecution result:\n{execution_context}\n\n"
            "Summarize the final answer in 2 concise bullet points. Preserve useful memory/state references without expanding them into long prose."
        )
        content = self._ask("You are a summarization agent.", prompt)
        payload = {
            "mt": "response",
            "a": "summarize",
            "in": {"task": _short(task), "execution": _short(execution_context)},
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


def _embedding_source_text(task: str, content: str) -> str:
    source = os.getenv("MEMORY_EMBEDDING_SOURCE", "task").strip().lower()
    if source == "content":
        return content
    if source == "hybrid":
        return f"{task}\n{_short(content, 320)}"
    return task


def _memory_refs(message: Message) -> list[str]:
    refs = message.payload.get("refs", []) if message.payload else []
    return [ref for ref in refs if isinstance(ref, str) and not ref.startswith("state_")]


def _agent_context(message: Message, limit: int = 420) -> str:
    if message.mode == "structured" and message.payload:
        context = {
            "action": message.payload.get("a"),
            "out": message.payload.get("out"),
            "refs": message.payload.get("refs", []),
            "state": message.payload.get("state", {}),
        }
        return _short(json.dumps(context, ensure_ascii=False, sort_keys=True), limit)
    return _short(message.content, limit)


def _protocol_tool_plan(task: str, refs: list[str]) -> dict[str, bool]:
    lowered = task.lower()
    wants_files = any(word in lowered for word in ("read ", "docs/", "markdown", "documents", "deployment", "technical"))
    wants_results = any(word in lowered for word in ("results.json", "ablation", "tokens", "elapsed", "metrics", "compare"))
    wants_table = any(word in lowered for word in ("table", "checklist", "files", "submission"))
    return {
        "read_markdown_files": wants_files or wants_table,
        "search_memory": False if refs else any(word in lowered for word in ("already analyzed", "discovered", "previous analysis")),
        "compute_metrics": wants_results,
        "make_table": wants_table or wants_results,
    }


def _compact_tool_result(result: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode != "structured":
        return result
    variables = result.get("variables", {}) if isinstance(result.get("variables"), dict) else {}
    compact_vars = {}
    for key in ("result", "metrics", "table", "files", "memory_hits"):
        if key not in variables:
            continue
        value = variables[key]
        if key == "files" and isinstance(value, list):
            compact_vars[key] = value[:8]
        elif key == "memory_hits" and isinstance(value, list):
            compact_vars[key] = [
                {
                    "memory_id": item.get("memory_id"),
                    "source_agent": item.get("source_agent"),
                    "tags": item.get("tags", [])[:4],
                }
                for item in value[:3]
                if isinstance(item, dict)
            ]
        elif isinstance(value, str):
            compact_vars[key] = _short(value, 360)
        else:
            compact_vars[key] = value
    return {
        "ok": bool(result.get("ok")),
        "stdout": _short(str(result.get("stdout", "")), 420),
        "variables": compact_vars,
        "error": _short(str(result.get("error") or ""), 180) if result.get("error") else None,
    }


def _tool_evidence_text(result: dict[str, Any]) -> str:
    variables = result.get("variables", {}) if isinstance(result.get("variables"), dict) else {}
    data = variables.get("result", {}) if isinstance(variables.get("result"), dict) else {}
    read_files = data.get("read_files") or variables.get("read_files") or []
    missing_files = data.get("missing_files") or variables.get("missing_files") or []
    json_fields = data.get("json_fields") or variables.get("json_fields") or {}
    table_preview = data.get("table_preview") or variables.get("table") or ""
    parts = []
    if read_files:
        parts.append("read_files=" + ", ".join(str(path) for path in read_files[:5]))
    if missing_files:
        parts.append("missing_files=" + ", ".join(str(path) for path in missing_files[:5]))
    if isinstance(json_fields, dict) and json_fields:
        field_parts = []
        for path, fields in list(json_fields.items())[:3]:
            if isinstance(fields, list):
                field_parts.append(f"{path}: {', '.join(str(field) for field in fields[:6])}")
            else:
                field_parts.append(f"{path}: {fields}")
        parts.append("json_fields=" + "; ".join(field_parts))
    if table_preview:
        parts.append("table=" + _short(str(table_preview), 360))
    if result.get("error"):
        parts.append("tool_error=" + _short(str(result.get("error")), 160))
    return _short(" | ".join(parts), 700)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

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


class VerifierAgent(BaseAgent):
    def verify(self, task: str, plan: Message, summary: Message, ctx: AgentContext) -> Message:
        system_prompt = "You are a quality assurance verifier agent. Assess if the summarization successfully completes the user task."
        plan_context = _agent_context(plan, limit=360 if ctx.mode == "structured" else 1000)
        summary_context = _agent_context(summary, limit=360 if ctx.mode == "structured" else 1000)
        user_prompt = (
            f"Original Task:\n{task}\n\n"
            f"Plan Proposed:\n{plan_context}\n\n"
            f"Final Summary:\n{summary_context}\n\n"
            "Assess whether the final summary successfully addresses the task. "
            "You must output a JSON response in the following format:\n"
            "{\n"
            '  "approved": true or false,\n'
            '  "feedback": "Your detailed feedback here if rejected, or empty if approved"\n'
            "}"
        )
        content = self._ask(system_prompt, user_prompt)
        
        approved = True
        feedback = ""
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(content)
            approved = bool(data.get("approved", True))
            feedback = str(data.get("feedback", ""))
        except Exception:
            if "false" in content.lower():
                approved = False
                feedback = "Failed to satisfy criteria."
            else:
                approved = True
                feedback = ""

        payload = {
            "mt": "response",
            "a": "verify",
            "in": {"task": _short(task), "summary": _short(summary_context)},
            "out": f"approved={approved} feedback={_short(feedback)}",
            "refs": [summary.message_id],
            "state": {
                "approved": approved,
                "feedback": feedback
            }
        }
        return _message(ctx, self.name, "orchestrator", content, payload, parent_id=summary.message_id)

    @property
    def capabilities(self) -> list[str]:
        return ["verification", "llm_chat"]




class RouterAgent(BaseAgent):
    def route(self, task: str, ctx: AgentContext) -> Message:
        if extract_file_paths(task):
            route_choice = "full_pipeline"
            payload = {
                "mt": "response",
                "a": "route",
                "in": {"task": _short(task)},
                "out": f"route={route_choice}",
                "refs": [],
                "state": {
                    "route": route_choice,
                    "reason": "explicit_file_path"
                }
            }
            return _message(ctx, self.name, "orchestrator", "Route to full pipeline because task names concrete files.", payload)

        system_prompt = (
            "You are a task routing agent. Your job is to analyze the user's task and decide the optimal execution path.\n"
            "Available routes:\n"
            "1. 'full_pipeline': For complex tasks requiring research, planning, database memory search, and writing Python code.\n"
            "2. 'direct_summarize': For simple informational or conversational questions that do not need code execution or planning."
        )
        user_prompt = (
            f"User Task:\n{task}\n\n"
            "Analyze the task complexity and select the most appropriate route. "
            "Output your decision in JSON format:\n"
            "{\n"
            '  "route": "full_pipeline" | "direct_summarize",\n'
            '  "reason": "Brief explanation of your routing choice"\n'
            "}"
        )
        content = self._ask(system_prompt, user_prompt)
        route_choice = "full_pipeline"
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(content)
            route_choice = str(data.get("route", "full_pipeline"))
        except Exception:
            if "direct_summarize" in content:
                route_choice = "direct_summarize"

        payload = {
            "mt": "response",
            "a": "route",
            "in": {"task": _short(task)},
            "out": f"route={route_choice}",
            "refs": [],
            "state": {
                "route": route_choice
            }
        }
        return _message(ctx, self.name, "orchestrator", content, payload)

    @property
    def capabilities(self) -> list[str]:
        return ["routing", "llm_chat"]


class SecurityReviewerAgent(BaseAgent):
    def review(self, task: str, code: str, ctx: AgentContext) -> Message:
        system_prompt = (
            "You are a security reviewer agent. Inspect the proposed Python code before execution in the sandbox. "
            "Check for any dangerous operations, such as deleting system files, running commands via subprocess, opening raw sockets, "
            "or potential infinite loops."
        )
        user_prompt = (
            f"Task: {task}\n\n"
            f"Proposed Python Code:\n```python\n{code}\n```\n\n"
            "Assess whether the code is safe to execute. Output your assessment in JSON format:\n"
            "{\n"
            '  "approved": true or false,\n'
            '  "feedback": "Detailed explanation of safety issues if rejected, or empty if approved"\n'
            "}"
        )
        content = self._ask(system_prompt, user_prompt)
        approved = True
        feedback = ""
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(content)
            approved = bool(data.get("approved", True))
            feedback = str(data.get("feedback", ""))
        except Exception:
            if "false" in content.lower():
                approved = False
                feedback = "Failed security audit."
            else:
                approved = True
                feedback = ""

        # Simple static checks as defense-in-depth
        blocked_keywords = ["os.system", "shutil.rmtree", "eval(", "exec("]
        for kw in blocked_keywords:
            if kw in code:
                approved = False
                feedback = f"Security Violation: usage of unsafe keyword '{kw}' detected."
                break

        payload = {
            "mt": "response",
            "a": "security_review",
            "in": {"code_len": len(code)},
            "out": f"approved={approved} feedback={_short(feedback)}",
            "refs": [],
            "state": {
                "approved": approved,
                "feedback": feedback
            }
        }
        return _message(ctx, self.name, "orchestrator", content, payload)

    @property
    def capabilities(self) -> list[str]:
        return ["security_review", "llm_chat"]


class DebuggerAgent(BaseAgent):
    def debug(self, task: str, code: str, error: str, stdout: str, ctx: AgentContext) -> Message:
        system_prompt = (
            "You are an expert debugger agent. Analyze code execution failures, tracebacks, and stdout, "
            "and suggest precise corrections or a clean code patch."
        )
        user_prompt = (
            f"Task: {task}\n\n"
            f"Code Attempted:\n```python\n{code}\n```\n\n"
            f"Execution Error:\n{error}\n\n"
            f"Execution Output (Stdout):\n{stdout}\n\n"
            "Analyze the root cause of the error. Output a JSON containing:\n"
            "{\n"
            '  "explanation": "Brief explanation of the bug",\n'
            '  "correction": "Clear step-by-step instructions or the modified code snippet to fix the issue"\n'
            "}"
        )
        content = self._ask(system_prompt, user_prompt)
        explanation = ""
        correction = ""
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(content)
            explanation = str(data.get("explanation", ""))
            correction = str(data.get("correction", ""))
        except Exception:
            explanation = "Runtime or compilation error."
            correction = content

        payload = {
            "mt": "response",
            "a": "debug",
            "in": {"error": _short(error)},
            "out": _short(explanation),
            "refs": [],
            "state": {
                "explanation": explanation,
                "correction": correction
            }
        }
        return _message(ctx, self.name, "executor", content, payload)

    @property
    def capabilities(self) -> list[str]:
        return ["debugging", "llm_chat"]


class MemoryArchivistAgent(BaseAgent):
    def archive(self, ctx: AgentContext) -> Message:
        system_prompt = (
            "You are a memory archivist agent. Your role is to examine the memory records "
            "in the database, clean up duplicates, build semantic connections (links), and refine summaries."
        )
        
        records = ctx.memory._all_records()
        if not records:
            out_msg = "No memory records to archive."
            payload = {
                "mt": "response",
                "a": "archive",
                "in": {},
                "out": out_msg,
                "refs": []
            }
            return _message(ctx, self.name, "user", out_msg, payload)
        
        formatted_records = []
        for r in records:
            formatted_records.append({
                "id": r.memory_id,
                "topic": r.task_topic,
                "summary": r.summary,
                "tags": r.tags,
                "links": r.links
            })
            
        user_prompt = (
            f"Here are the existing memory records in the shared memory database:\n"
            f"{json.dumps(formatted_records, ensure_ascii=False, indent=2)}\n\n"
            "Analyze these records and suggest connection links between related tasks. "
            "Also identify any redundant or outdated records. Output your suggestions in JSON format:\n"
            "{\n"
            '  "links_to_add": [\n'
            '     {"source": "id1", "target": "id2"}\n'
            '  ],\n'
            '  "redundant_ids": ["id3"]\n'
            "}"
        )
        content = self._ask(system_prompt, user_prompt)
        links_to_add = []
        redundant_ids = []
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(content)
            links_to_add = list(data.get("links_to_add", []))
            redundant_ids = list(data.get("redundant_ids", []))
        except Exception:
            pass

        actions_taken = []
        for link in links_to_add:
            src = link.get("source")
            tgt = link.get("target")
            if src and tgt:
                try:
                    ctx.memory._upsert_links(src, [tgt])
                    row = ctx.memory._conn.execute("SELECT links FROM memories WHERE memory_id = ?", (src,)).fetchone()
                    if row:
                        existing_links = list(json.loads(row[0]))
                        if tgt not in existing_links:
                            existing_links.append(tgt)
                            ctx.memory._conn.execute(
                                "UPDATE memories SET links = ? WHERE memory_id = ?",
                                (json.dumps(existing_links), src)
                            )
                            ctx.memory._conn.commit()
                    actions_taken.append(f"Linked {src} -> {tgt}")
                except Exception:
                    pass
                    
        for rid in redundant_ids:
            try:
                ctx.memory._conn.execute("DELETE FROM memories WHERE memory_id = ?", (rid,))
                ctx.memory._conn.execute("DELETE FROM memory_links WHERE source_memory_id = ? OR target_memory_id = ?", (rid, rid))
                ctx.memory._conn.commit()
                actions_taken.append(f"Deleted {rid}")
            except Exception:
                pass
                
        out_msg = ", ".join(actions_taken) if actions_taken else "Memory database is optimized."
        payload = {
            "mt": "response",
            "a": "archive",
            "in": {"records_analyzed": len(records)},
            "out": out_msg,
            "refs": [r.memory_id for r in records],
            "state": {
                "links_added": len(links_to_add),
                "deleted": len(redundant_ids),
                "actions": actions_taken
            }
        }
        return _message(ctx, self.name, "user", out_msg, payload)

    @property
    def capabilities(self) -> list[str]:
        return ["memory_consolidation", "llm_chat"]
