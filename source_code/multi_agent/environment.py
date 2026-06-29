from __future__ import annotations

import os
import platform
from pathlib import Path


def collect_environment() -> dict[str, str]:
    os_release = _read_os_release()
    return {
        "os_pretty_name": os_release.get("PRETTY_NAME", platform.platform()),
        "os_id": os_release.get("ID", ""),
        "os_version_id": os_release.get("VERSION_ID", ""),
        "python_version": platform.python_version(),
        "llm_backend": os.getenv("LLM_BACKEND", "required"),
        "llm_model": os.getenv("LLM_MODEL", "required"),
        "embedding_backend": os.getenv("EMBEDDING_BACKEND", "hash"),
        "embedding_model": os.getenv("OLLAMA_EMBED_MODEL", os.getenv("EMBEDDING_MODEL", "hash")),
        "orchestrator": os.getenv("ORCHESTRATOR", "sequential"),
        "token_count_method": os.getenv("TOKEN_COUNT_METHOD", "unicode_heuristic"),
        "memory_path": os.getenv("MEMORY_PATH", "data/memory.sqlite"),
        "state_path": os.getenv("STATE_PATH", "data/state.sqlite"),
        "state_backend": os.getenv("STATE_BACKEND", "shared_memory"),
        "codeact_sandbox": os.getenv("CODEACT_SANDBOX", "subprocess"),
        "orchestrator_max_retries": os.getenv("ORCHESTRATOR_MAX_RETRIES", "2"),
        "memory_archivist_enabled": os.getenv("MEMORY_ARCHIVIST_ENABLED", "0"),
        "embedding_cache_enabled": os.getenv("EMBEDDING_CACHE_ENABLED", "1"),
        "memory_embedding_source": os.getenv("MEMORY_EMBEDDING_SOURCE", "task"),
        "memory_search_limit": os.getenv("MEMORY_SEARCH_LIMIT", "1"),
        "memory_graph_max_candidates": os.getenv("MEMORY_GRAPH_MAX_CANDIDATES", "16"),
        "memory_write_policy": os.getenv("MEMORY_WRITE_POLICY", "topic_once"),
        "memory_write_on_reuse": os.getenv("MEMORY_WRITE_ON_REUSE", "1"),
    }


def _read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values
