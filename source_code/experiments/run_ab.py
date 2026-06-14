from __future__ import annotations

import json
import os
from pathlib import Path

from multi_agent.runner import ExperimentRunner
from multi_agent.tasks import DEFAULT_TASKS


def main() -> None:
    output_path = Path("reports/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runner = ExperimentRunner()
    rounds = int(os.getenv("EXPERIMENT_ROUNDS", "10"))
    results = runner.run_ab(tasks=DEFAULT_TASKS, rounds=rounds)

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"\nFull report written to {output_path}")


if __name__ == "__main__":
    main()
