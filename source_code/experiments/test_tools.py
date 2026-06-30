from __future__ import annotations

import json

from multi_agent.memory import SharedMemory
from multi_agent.tools import CodeActExecutor, ToolRegistry, build_codeact_for_task


def main() -> None:
    memory = SharedMemory(db_path="/tmp/llm_tool_smoke.sqlite", reset=True)
    memory.add(
        source_agent="test",
        task_topic="tool smoke test",
        summary="Reusable memory for testing search and table generation tools.",
        tags=["tool", "memory"],
    )
    registry = ToolRegistry(memory=memory)
    executor = CodeActExecutor()

    run_basic_tool_smoke(executor, registry)
    run_file_path_tool_smoke(executor, registry)


def run_basic_tool_smoke(executor: CodeActExecutor, registry: ToolRegistry) -> None:
    result = executor.run(
        build_codeact_for_task("tool smoke test", "findings from retrieval and analysis"),
        context=registry.as_context(),
    )
    assert result.ok, result.error
    assert "result" in result.variables
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


def run_file_path_tool_smoke(executor: CodeActExecutor, registry: ToolRegistry) -> None:
    tool_task = (
        "Read docs/deployment.md and reports/results.json if available, "
        "then summarize expected output files. Also check reports/not_present.json if available."
    )
    result = executor.run(
        build_codeact_for_task(tool_task, "Need file-backed evidence."),
        context=registry.as_context(),
    )
    assert result.ok, result.error

    payload = result.variables.get("result", {})
    assert payload, result.variables
    assert "target_files" in payload
    assert "read_files" in payload
    assert "missing_files" in payload
    assert "file_records" in payload
    assert "table_preview" in payload
    assert payload["target_files"]
    assert "docs/deployment.md" in payload["target_files"]
    assert "docs/deployment.md" in payload["read_files"]
    assert "reports/not_present.json" in payload["missing_files"]
    assert payload["read_files"] or payload["missing_files"]
    assert payload["file_records"]
    assert "docs/deployment.md" in payload["table_preview"]

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
