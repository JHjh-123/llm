# Multi-Agent Communication Demo Report

- Generated at: 2026-06-14T17:24:17
- Rounds: 1
- LLM backend: ollama
- LLM model: qwen3:8b
- Embedding backend: ollama
- Embedding model: bge-m3
- Orchestrator: langgraph

## Demo Goal

展示多 Agent 在纯文本通信与结构化通信下的协作流程，并比较消息数、通信 token、耗时、记忆命中、非文本状态传递等指标。

## Requirement Coverage

| Requirement | Current Status | Evidence |
| --- | --- | --- |
| 3+ Agent collaboration | Done | Planner, Researcher, Executor, Summarizer |
| Text vs structured A/B | Done | Same tasks run in both modes |
| Formal protocol | Done | message_id, task_id, parent_id, handshake, capability discovery |
| LangGraph orchestration | Done when ORCHESTRATOR=langgraph | Metrics record orchestrator name |
| Shared memory | Done | SQLite memory store |
| Real embedding | Done when EMBEDDING_BACKEND=ollama | bge-m3 returns 1024-d vectors |
| Non-text state transfer | Done | embedding/state refs/tool result counted |
| CodeAct tools | Done | restricted Python tool execution |
| Reproducible metrics | Prototype done | JSON and Markdown reports generated |

## A/B Summary

| Metric | Text | Structured |
| --- | ---: | ---: |
| Runs | 2 | 2 |
| Avg messages | 4.0 | 4.0 |
| Avg chars | 3455.0 | 2459.0 |
| Avg approx tokens | 862.0 | 613.5 |
| Avg elapsed ms | 21102.385 | 10143.509 |
| Avg memory hits | 1.0 | 2.0 |
| Avg non-text transfers | 2.5 | 3.0 |
| Avg protocol events | 0.0 | 5.0 |
| Avg protocol approx tokens | 0.0 | 415.0 |

## Main Finding

- Structured application-message token delta: **-28.83%**
- Current conclusion: **有优势**
- Protocol negotiation overhead is reported separately from application messages.

## Agent Trace Example

- Task: Design a structured protocol for three agents to solve a document summarization task.

| From | To | Action | Refs | State Keys | Approx Tokens |
| --- | --- | --- | ---: | --- | ---: |
| planner | researcher | plan | 0 |  | 98 |
| researcher | executor | research | 1 |  | 139 |
| executor | summarizer | execute | 1 | emb_dim, tool, tool_ok, tool_result | 232 |
| summarizer | user | summarize | 1 |  | 139 |

## Remaining Work For Final Submission

- Run at least 10 rounds and report mean/variance.
- Add more task groups, especially related continuous tasks.
- Replace approximate token counting with model tokenizer counting.
- Strengthen sandboxing if CodeAct runs untrusted code.
- Verify the full system on openEuler 24.03-LTS-SP3.
