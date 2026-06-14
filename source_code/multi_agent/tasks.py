TASK_GROUPS = {
    "protocol_design": [
        "Design a structured protocol for three agents to solve a document summarization task.",
        "Reuse memory from the previous protocol design task and evaluate communication overhead.",
    ],
    "memory_reuse": [
        "Plan a shared memory schema for multi-agent research tasks with keyword, tag, and semantic retrieval.",
        "Use the previous memory schema plan to evaluate how memory hits reduce repeated analysis.",
    ],
    "tool_execution": [
        "Design a CodeAct tool workflow for reading local reports and computing compact metrics.",
        "Reuse the CodeAct workflow to summarize saved experiment results and transfer non-text state.",
    ],
}

DEFAULT_TASKS = [task for group in TASK_GROUPS.values() for task in group]
