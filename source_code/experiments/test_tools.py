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
    result = executor.run(
        build_codeact_for_task("tool smoke test", "findings from retrieval and analysis"),
        context=registry.as_context(),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
