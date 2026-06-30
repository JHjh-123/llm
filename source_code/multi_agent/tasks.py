TASK_GROUPS = {
    "document_analysis": [
        "Read docs/deployment.md and summarize the required environment, startup commands, and expected output files.",
        "Based on the deployment constraints already analyzed, produce a concise reproducibility checklist for running the experiment on openEuler.",
    ],
    "result_analysis": [
        "Read reports/results.json and reports/ablation_results.json if available, then compare text and structured modes on average tokens, elapsed time, memory hits, and non-text state transfers.",
        "Explain whether the structured mode reduces communication overhead, using the available experiment results as evidence.",
    ],
    "technical_review": [
        "Read docs/technical_description.md and summarize the system modules, including protocol, memory, state exchange, orchestration, and CodeAct tools.",
        "Assess which module contributes most to reducing communication cost and identify one remaining weakness in the current design.",
    ],
    "tool_execution": [
        "Find the Markdown files in the workspace and build a compact table describing the purpose of each file.",
        "Using the discovered project documents, generate a short final submission checklist with required source files, reports, and runtime evidence.",
    ],
}

DEFAULT_TASKS = [task for group in TASK_GROUPS.values() for task in group]
