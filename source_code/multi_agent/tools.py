from __future__ import annotations

import ast
import csv
import io
import json
import os
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}

BLOCKED_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.For,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
)


@dataclass
class ToolResult:
    ok: bool
    stdout: str
    variables: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodeActExecutor:
    """Tiny restricted Python action space for prototype tool execution."""

    def run(self, code: str, context: dict[str, Any] | None = None) -> ToolResult:
        try:
            tree = ast.parse(code, mode="exec")
            self._validate(tree, allowed_calls=set((context or {}).keys()))
            env: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
            if context:
                env.update(context)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exec(compile(tree, "<codeact>", "exec"), env, env)
            variables = {
                key: value
                for key, value in env.items()
                if not key.startswith("__") and key not in SAFE_BUILTINS and _json_safe(value)
            }
            return ToolResult(ok=True, stdout=stdout.getvalue(), variables=variables)
        except Exception as exc:
            return ToolResult(ok=False, stdout="", variables={}, error=f"{type(exc).__name__}: {exc}")

    def _validate(self, tree: ast.AST, allowed_calls: set[str]) -> None:
        for node in ast.walk(tree):
            if isinstance(node, BLOCKED_NODES):
                raise ValueError(f"Blocked Python construct: {type(node).__name__}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_BUILTINS and node.func.id not in allowed_calls:
                    raise ValueError(f"Blocked function call: {node.func.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("Dunder attribute access is blocked")


class ToolRegistry:
    def __init__(self, memory: Any | None = None, workspace_root: str | Path | None = None) -> None:
        self.memory = memory
        self.workspace_root = Path(workspace_root or os.getenv("TOOL_WORKSPACE") or _default_workspace()).resolve()

    def as_context(self) -> dict[str, Any]:
        return {
            "read_file": self.read_file,
            "search_files": self.search_files,
            "load_json": self.load_json,
            "load_csv": self.load_csv,
            "make_markdown_table": self.make_markdown_table,
            "compute_numeric_metrics": self.compute_numeric_metrics,
            "summarize_records": self.summarize_records,
            "search_memory": self.search_memory,
        }

    def read_file(self, path: str, max_chars: int = 4000) -> str:
        target = self._resolve(path)
        return target.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def search_files(self, pattern: str = "*.md", max_results: int = 20) -> list[str]:
        if ".." in Path(pattern).parts:
            raise ValueError("Parent path traversal is not allowed in search pattern")
        results = []
        for path in self.workspace_root.rglob(pattern):
            if path.is_file():
                results.append(str(path.relative_to(self.workspace_root)))
            if len(results) >= max_results:
                break
        return results

    def load_json(self, path: str, max_chars: int = 200000) -> Any:
        return json.loads(self.read_file(path, max_chars=max_chars))

    def load_csv(self, path: str, max_rows: int = 100) -> list[dict[str, str]]:
        text = self.read_file(path, max_chars=200000)
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= max_rows:
                break
        return rows

    def make_markdown_table(
        self,
        rows: list[dict[str, Any]],
        columns: list[str] | None = None,
        max_rows: int = 20,
    ) -> str:
        if not rows:
            return ""
        selected_columns = columns or list(rows[0].keys())
        lines = [
            "| " + " | ".join(selected_columns) + " |",
            "| " + " | ".join(["---"] * len(selected_columns)) + " |",
        ]
        for row in rows[:max_rows]:
            values = [_cell(row.get(column, "")) for column in selected_columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def compute_numeric_metrics(self, values: list[int | float]) -> dict[str, float | int | None]:
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        if not numeric:
            return {"count": 0, "mean": None, "min": None, "max": None, "pstdev": None}
        return {
            "count": len(numeric),
            "mean": round(mean(numeric), 4),
            "min": min(numeric),
            "max": max(numeric),
            "pstdev": round(pstdev(numeric), 4) if len(numeric) > 1 else 0.0,
        }

    def summarize_records(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        columns = sorted({key for row in rows for key in row.keys()})
        return {"row_count": len(rows), "columns": columns}

    def search_memory(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        if self.memory is None:
            return []
        records = self.memory.search(query, limit=limit)
        return [
            {
                "memory_id": record.memory_id,
                "source_agent": record.source_agent,
                "task_topic": record.task_topic,
                "summary": record.summary[:240],
                "tags": record.tags,
            }
            for record in records
        ]

    def _resolve(self, path: str) -> Path:
        target = (self.workspace_root / path).resolve()
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"Path is outside tool workspace: {path}") from exc
        if not target.is_file():
            raise FileNotFoundError(path)
        return target


def build_codeact_for_task(task: str, findings: str) -> str:
    return "\n".join(
        [
            f"task = {task!r}",
            f"findings = {findings!r}",
            "files = search_files('*.md', 5)",
            "memory_hits = search_memory(task, 2)",
            "metrics = compute_numeric_metrics([len(task), len(findings)])",
            "table = make_markdown_table([",
            "    {'metric': 'task_chars', 'value': len(task)},",
            "    {'metric': 'finding_chars', 'value': len(findings)},",
            "    {'metric': 'memory_hits', 'value': len(memory_hits)},",
            "])",
            "result = {",
            "    'task_words': len(task.split()),",
            "    'finding_chars': len(findings),",
            "    'workspace_files_seen': len(files),",
            "    'memory_tool_hits': len(memory_hits),",
            "    'metrics': metrics,",
            "    'table_preview': table[:240],",
            "    'reusable': True,",
            "}",
            "print(result)",
        ]
    )


def _json_safe(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, list, tuple, dict))


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")[:120]


def _default_workspace() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "source_code" and cwd.parent.exists():
        return cwd.parent
    return cwd
