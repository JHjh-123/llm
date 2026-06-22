from __future__ import annotations

import ast
import csv
import io
import json
import os
import subprocess
import sys
import textwrap
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
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
)

SUBPROCESS_BLOCKED_NODES = (
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.With,
)

SAFE_IMPORT_MODULES = {
    "collections",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
}


@dataclass
class ToolResult:
    ok: bool
    stdout: str
    variables: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CodeActExecutor:
    """Restricted Python action space for CodeAct-style tool execution."""

    def run(self, code: str, context: dict[str, Any] | None = None) -> ToolResult:
        mode = os.getenv("CODEACT_SANDBOX", "subprocess").lower()
        if mode == "subprocess":
            return self._run_subprocess(code, context=context)
        return self._run_inprocess(code, context=context)

    def _run_inprocess(self, code: str, context: dict[str, Any] | None = None) -> ToolResult:
        try:
            tree = ast.parse(code, mode="exec")
            self._validate(tree, allowed_calls=set((context or {}).keys()), subprocess_mode=False)
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

    def _run_subprocess(self, code: str, context: dict[str, Any] | None = None) -> ToolResult:
        try:
            tree = ast.parse(code, mode="exec")
            self._validate(tree, allowed_calls=set((context or {}).keys()), subprocess_mode=True)
            payload = json.dumps(
                {
                    "code": code,
                    "workspace_root": str(_default_workspace()),
                    "memory_records": _memory_records_from_context(context),
                    "cpu_seconds": int(os.getenv("CODEACT_CPU_SECONDS", "2")),
                    "memory_mb": int(os.getenv("CODEACT_MEMORY_MB", "512")),
                },
                ensure_ascii=False,
            ).encode("utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _SUBPROCESS_RUNTIME],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(os.getenv("CODEACT_TIMEOUT_SECONDS", "5")),
                check=False,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.decode("utf-8", errors="replace").strip()
                return ToolResult(ok=False, stdout="", variables={}, error=f"Subprocess failed: {stderr}")
            result = json.loads(completed.stdout.decode("utf-8"))
            return ToolResult(
                ok=bool(result.get("ok")),
                stdout=str(result.get("stdout", "")),
                variables=dict(result.get("variables", {})),
                error=result.get("error"),
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(ok=False, stdout="", variables={}, error=f"TimeoutExpired: {exc}")
        except Exception as exc:
            return ToolResult(ok=False, stdout="", variables={}, error=f"{type(exc).__name__}: {exc}")

    def _validate(self, tree: ast.AST, allowed_calls: set[str], subprocess_mode: bool) -> None:
        blocked_nodes = SUBPROCESS_BLOCKED_NODES if subprocess_mode else BLOCKED_NODES
        for node in ast.walk(tree):
            if isinstance(node, blocked_nodes):
                raise ValueError(f"Blocked Python construct: {type(node).__name__}")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if not subprocess_mode:
                    raise ValueError(f"Blocked Python construct: {type(node).__name__}")
                module_names = _import_module_names(node)
                blocked = [name for name in module_names if name.split(".")[0] not in SAFE_IMPORT_MODULES]
                if blocked:
                    raise ValueError(f"Blocked import module: {', '.join(blocked)}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "compile", "open", "input", "__import__"}:
                    raise ValueError(f"Blocked function call: {node.func.id}")
                if not subprocess_mode and node.func.id not in SAFE_BUILTINS and node.func.id not in allowed_calls:
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
                "keywords": record.keywords[:8],
                "links": record.links[:5],
                "access_count": record.access_count,
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


_SUBPROCESS_RUNTIME = textwrap.dedent(
    r"""
    import ast
    import builtins
    import csv
    import io
    import json
    import os
    import sys
    import traceback
    from contextlib import redirect_stdout
    from pathlib import Path
    from statistics import mean, pstdev

    SAFE_IMPORT_MODULES = {"collections", "itertools", "json", "math", "re", "statistics"}
    SAFE_BUILTIN_NAMES = [
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "isinstance", "len", "list", "map", "max", "min", "print", "range", "round",
        "set", "sorted", "str", "sum", "tuple", "zip",
    ]

    def _set_limits(cpu_seconds, memory_mb):
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 1))
            bytes_limit = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        except Exception:
            pass

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root not in SAFE_IMPORT_MODULES:
            raise ImportError(f"import blocked: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    def _json_safe(value):
        return value is None or isinstance(value, (str, int, float, bool, list, tuple, dict))

    def _cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")[:120]

    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    _set_limits(payload.get("cpu_seconds", 2), payload.get("memory_mb", 512))
    workspace_root = Path(payload["workspace_root"]).resolve()
    memory_records = list(payload.get("memory_records", []))

    def _resolve(path):
        target = (workspace_root / path).resolve()
        try:
            target.relative_to(workspace_root)
        except ValueError as exc:
            raise ValueError(f"Path is outside tool workspace: {path}") from exc
        if not target.is_file():
            raise FileNotFoundError(path)
        return target

    def read_file(path, max_chars=4000):
        return _resolve(path).read_text(encoding="utf-8", errors="replace")[:max_chars]

    def search_files(pattern="*.md", max_results=20):
        if ".." in Path(pattern).parts:
            raise ValueError("Parent path traversal is not allowed in search pattern")
        results = []
        for path in workspace_root.rglob(pattern):
            if path.is_file():
                results.append(str(path.relative_to(workspace_root)))
            if len(results) >= max_results:
                break
        return results

    def load_json(path, max_chars=200000):
        return json.loads(read_file(path, max_chars=max_chars))

    def load_csv(path, max_rows=100):
        text = read_file(path, max_chars=200000)
        reader = csv.DictReader(io.StringIO(text))
        rows = []
        for row in reader:
            rows.append(dict(row))
            if len(rows) >= max_rows:
                break
        return rows

    def make_markdown_table(rows, columns=None, max_rows=20):
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

    def compute_numeric_metrics(values):
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

    def summarize_records(rows):
        columns = sorted({key for row in rows for key in row.keys()})
        return {"row_count": len(rows), "columns": columns}

    def search_memory(query, limit=3):
        query_terms = {term.lower() for term in str(query).split()}
        scored = []
        for record in memory_records:
            haystack = " ".join(
                [
                    str(record.get("task_topic", "")),
                    str(record.get("summary", "")),
                    " ".join(record.get("tags", [])),
                    " ".join(record.get("keywords", [])),
                ]
            ).lower()
            score = sum(1 for term in query_terms if term and term in haystack)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}
    safe_builtins["__import__"] = _safe_import
    env = {
        "__builtins__": safe_builtins,
        "read_file": read_file,
        "search_files": search_files,
        "load_json": load_json,
        "load_csv": load_csv,
        "make_markdown_table": make_markdown_table,
        "compute_numeric_metrics": compute_numeric_metrics,
        "summarize_records": summarize_records,
        "search_memory": search_memory,
    }

    try:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exec(compile(payload["code"], "<codeact-subprocess>", "exec"), env, env)
        variables = {
            key: value
            for key, value in env.items()
            if not key.startswith("__") and key not in {
                "read_file", "search_files", "load_json", "load_csv", "make_markdown_table",
                "compute_numeric_metrics", "summarize_records", "search_memory"
            } and _json_safe(value)
        }
        print(json.dumps({"ok": True, "stdout": stdout.getvalue(), "variables": variables, "error": None}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "stdout": "", "variables": {}, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
    """
)


def _json_safe(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool, list, tuple, dict))


def _import_module_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return [node.module or ""]


def _memory_records_from_context(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not context or "search_memory" not in context:
        return []
    registry = getattr(context["search_memory"], "__self__", None)
    memory = getattr(registry, "memory", None)
    if memory is None:
        return []
    try:
        records = memory.to_dict()
    except Exception:
        return []
    safe_records = []
    for record in records:
        safe_records.append(
            {
                "memory_id": record.get("memory_id"),
                "source_agent": record.get("source_agent"),
                "task_topic": record.get("task_topic"),
                "summary": str(record.get("summary", ""))[:500],
                "tags": record.get("tags", []),
                "keywords": record.get("keywords", []),
                "links": record.get("links", []),
                "access_count": record.get("access_count", 0),
            }
        )
    return safe_records


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")[:120]


def _default_workspace() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "source_code" and cwd.parent.exists():
        return cwd.parent
    return cwd
