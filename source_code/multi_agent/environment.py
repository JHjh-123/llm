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
