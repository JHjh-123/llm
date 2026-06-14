from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from multi_agent.embedding import EmbeddingProvider, build_embedding_provider_from_env


@dataclass
class MemoryRecord:
    memory_id: str
    source_agent: str
    created_at: float
    task_topic: str
    summary: str
    tags: list[str]
    embedding: list[float]


class SharedMemory:
    def __init__(
        self,
        db_path: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reset: bool = False,
    ) -> None:
        self.db_path = Path(db_path or os.getenv("MEMORY_PATH", "data/memory.sqlite"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.db_path.exists():
            self.db_path.unlink()
        self.embedding_provider = embedding_provider or build_embedding_provider_from_env()
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def add(
        self,
        source_agent: str,
        task_topic: str,
        summary: str,
        tags: list[str],
        embedding: list[float] | None = None,
    ) -> MemoryRecord:
        vector = embedding or self.embed(summary)
        memory_id = hashlib.sha1(f"{source_agent}:{task_topic}:{summary}".encode("utf-8")).hexdigest()[:12]
        record = MemoryRecord(
            memory_id=memory_id,
            source_agent=source_agent,
            created_at=time.time(),
            task_topic=task_topic,
            summary=summary,
            tags=tags,
            embedding=vector,
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (memory_id, source_agent, created_at, task_topic, summary, tags, embedding, embedding_dim)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.memory_id,
                record.source_agent,
                record.created_at,
                record.task_topic,
                record.summary,
                json.dumps(record.tags, ensure_ascii=False),
                json.dumps(record.embedding),
                len(record.embedding),
            ),
        )
        self._conn.commit()
        return record

    def search(self, query: str, limit: int = 3) -> list[MemoryRecord]:
        query_embedding = self.embed(query)
        query_terms = _terms(query)
        scored: list[tuple[float, MemoryRecord]] = []

        for record in self._all_records():
            tag_score = len(query_terms.intersection(record.tags)) * 0.2
            text_terms = _terms(f"{record.task_topic} {record.summary}")
            text_score = len(query_terms.intersection(text_terms)) * 0.05
            vector_score = _cosine(query_embedding, record.embedding)
            score = vector_score + tag_score + text_score
            if score > 0.15:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def embed(self, text: str) -> list[float]:
        return self.embedding_provider.embed(text)

    def to_dict(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self._all_records()]

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                source_agent TEXT NOT NULL,
                created_at REAL NOT NULL,
                task_topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags TEXT NOT NULL,
                embedding TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_agent)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
        self._conn.commit()

    def _all_records(self) -> list[MemoryRecord]:
        rows = self._conn.execute(
            """
            SELECT memory_id, source_agent, created_at, task_topic, summary, tags, embedding
            FROM memories
            ORDER BY created_at ASC
            """
        ).fetchall()
        return [_row_to_record(row) for row in rows]


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=str(row["memory_id"]),
        source_agent=str(row["source_agent"]),
        created_at=float(row["created_at"]),
        task_topic=str(row["task_topic"]),
        summary=str(row["summary"]),
        tags=list(json.loads(row["tags"])),
        embedding=[float(value) for value in json.loads(row["embedding"])],
    )


def _terms(text: str) -> set[str]:
    return {token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()}


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
