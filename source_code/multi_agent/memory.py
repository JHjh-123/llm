from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from multi_agent.embedding import EmbeddingProvider, build_embedding_provider_from_env

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


@dataclass
class MemoryRecord:
    memory_id: str
    source_agent: str
    created_at: float
    task_topic: str
    summary: str
    tags: list[str]
    embedding: list[float]
    keywords: list[str]
    links: list[str]
    access_count: int = 0


class SharedMemory:
    def __init__(
        self,
        db_path: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reset: bool = False,
    ) -> None:
        self.db_type = os.getenv("DATABASE_TYPE", "sqlite").lower()
        self.postgres_url = os.getenv("DATABASE_URL", "postgresql://postgres:123456@localhost:5432/multi_agent")
        self.embedding_provider = embedding_provider or build_embedding_provider_from_env()
        
        connected = False
        if self.db_type == "postgres":
            if HAS_PSYCOPG2:
                try:
                    self._conn = psycopg2.connect(self.postgres_url)
                    self._conn.autocommit = True
                    connected = True
                    if reset:
                        try:
                            with self._conn.cursor() as cur:
                                cur.execute("DROP TABLE IF EXISTS memories CASCADE")
                                cur.execute("DROP TABLE IF EXISTS memory_links CASCADE")
                        except Exception:
                            pass
                except Exception as e:
                    print(f"Warning: Failed to connect to PostgreSQL: {e}. Falling back to SQLite.")
                    self.db_type = "sqlite"
            else:
                print("Warning: psycopg2-binary package not found. Falling back to SQLite.")
                self.db_type = "sqlite"

        if not connected:
            self.db_type = "sqlite"
            self.db_path = Path(db_path or os.getenv("MEMORY_PATH", "data/memory.sqlite"))
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            if reset and self.db_path.exists():
                self.db_path.unlink()
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
        graph_enabled = _env_bool("MEMORY_GRAPH_ENABLED", default=True)
        keywords = sorted(_terms(f"{task_topic} {summary}").union(tags))[:16] if graph_enabled else []
        links = self._related_memory_ids(vector, keywords, exclude_id=memory_id) if graph_enabled else []
        record = MemoryRecord(
            memory_id=memory_id,
            source_agent=source_agent,
            created_at=time.time(),
            task_topic=task_topic,
            summary=summary,
            tags=tags,
            embedding=vector,
            keywords=keywords,
            links=links,
        )

        if self.db_type == "postgres":
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memories
                    (
                        memory_id, source_agent, created_at, task_topic, summary,
                        tags, embedding, embedding_dim, keywords, links, access_count
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE((SELECT access_count FROM memories WHERE memory_id = %s), 0))
                    ON CONFLICT (memory_id) DO UPDATE SET
                        source_agent = EXCLUDED.source_agent,
                        created_at = EXCLUDED.created_at,
                        task_topic = EXCLUDED.task_topic,
                        summary = EXCLUDED.summary,
                        tags = EXCLUDED.tags,
                        embedding = EXCLUDED.embedding,
                        embedding_dim = EXCLUDED.embedding_dim,
                        keywords = EXCLUDED.keywords,
                        links = EXCLUDED.links
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
                        json.dumps(record.keywords, ensure_ascii=False),
                        json.dumps(record.links, ensure_ascii=False),
                        record.memory_id,
                    ),
                )
        else:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (
                    memory_id, source_agent, created_at, task_topic, summary,
                    tags, embedding, embedding_dim, keywords, links, access_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT access_count FROM memories WHERE memory_id = ?), 0))
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
                    json.dumps(record.keywords, ensure_ascii=False),
                    json.dumps(record.links, ensure_ascii=False),
                    record.memory_id,
                ),
            )
            self._conn.commit()

        self._upsert_links(record.memory_id, record.links)
        return record

    def search(self, query: str | list[float], limit: int = 3) -> list[MemoryRecord]:
        if isinstance(query, (list, tuple)):
            query_embedding = list(query)
            query_terms = set()
        elif isinstance(query, str):
            query_embedding = self.embed(query)
            query_terms = _terms(query)
        else:
            query_embedding = list(query)
            query_terms = set()

        scored: list[tuple[float, MemoryRecord]] = []
        all_records = self._all_records()
        if not all_records:
            return []

        if HAS_NUMPY:
            embeddings = np.array([r.embedding for r in all_records], dtype=np.float64)
            q_emb = np.array(query_embedding, dtype=np.float64)
            norms = np.linalg.norm(embeddings, axis=1)
            q_norm = np.linalg.norm(q_emb)
            norms[norms == 0.0] = 1.0
            if q_norm == 0.0:
                q_norm = 1.0
            vector_scores = (np.dot(embeddings, q_emb) / (norms * q_norm)).tolist()
        else:
            vector_scores = [_cosine(query_embedding, record.embedding) for record in all_records]

        for idx, record in enumerate(all_records):
            tag_score = len(query_terms.intersection(record.tags)) * 0.2
            graph_enabled = _env_bool("MEMORY_GRAPH_ENABLED", default=True)
            keyword_score = len(query_terms.intersection(record.keywords)) * 0.1 if graph_enabled else 0.0
            text_terms = _terms(f"{record.task_topic} {record.summary}")
            text_score = len(query_terms.intersection(text_terms)) * 0.05
            vector_score = vector_scores[idx]
            link_score = len(record.links) * 0.01 if graph_enabled else 0.0
            score = vector_score + tag_score + keyword_score + text_score + link_score
            if score > 0.15:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        records = [record for _, record in scored[:limit]]
        self._mark_accessed([record.memory_id for record in records])
        return records

    def embed(self, text: str) -> list[float]:
        return self.embedding_provider.embed(text)

    def to_dict(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self._all_records()]

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        if self.db_type == "postgres":
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id TEXT PRIMARY KEY,
                        source_agent TEXT NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL,
                        task_topic TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        tags TEXT NOT NULL,
                        embedding TEXT NOT NULL,
                        embedding_dim INTEGER NOT NULL,
                        keywords TEXT NOT NULL DEFAULT '[]',
                        links TEXT NOT NULL DEFAULT '[]',
                        access_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_links (
                        source_memory_id TEXT NOT NULL,
                        target_memory_id TEXT NOT NULL,
                        created_at DOUBLE PRECISION NOT NULL,
                        PRIMARY KEY (source_memory_id, target_memory_id)
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_agent)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_memory_id)")
                self._ensure_column("memories", "keywords", "TEXT NOT NULL DEFAULT '[]'")
                self._ensure_column("memories", "links", "TEXT NOT NULL DEFAULT '[]'")
                self._ensure_column("memories", "access_count", "INTEGER NOT NULL DEFAULT 0")
        else:
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
                    embedding_dim INTEGER NOT NULL,
                    keywords TEXT NOT NULL DEFAULT '[]',
                    links TEXT NOT NULL DEFAULT '[]',
                    access_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._ensure_column("memories", "keywords", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("memories", "links", "TEXT NOT NULL DEFAULT '[]'")
            self._ensure_column("memories", "access_count", "INTEGER NOT NULL DEFAULT 0")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_links (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id)
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_agent)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links(source_memory_id)")
            self._conn.commit()

    def _all_records(self) -> list[MemoryRecord]:
        query = """
            SELECT memory_id, source_agent, created_at, task_topic, summary, tags, embedding, keywords, links, access_count
            FROM memories
            ORDER BY created_at ASC
        """
        if self.db_type == "postgres":
            with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
        else:
            rows = self._conn.execute(query).fetchall()
        return [_row_to_record(row) for row in rows]

    def _related_memory_ids(self, embedding: list[float], keywords: list[str], exclude_id: str) -> list[str]:
        scored = []
        keyword_set = set(keywords)
        all_records = self._all_records()
        records_to_score = [r for r in all_records if r.memory_id != exclude_id and len(r.embedding) == len(embedding)]
        if not records_to_score:
            return []

        if HAS_NUMPY:
            embeddings = np.array([r.embedding for r in records_to_score], dtype=np.float64)
            q_emb = np.array(embedding, dtype=np.float64)
            norms = np.linalg.norm(embeddings, axis=1)
            q_norm = np.linalg.norm(q_emb)
            norms[norms == 0.0] = 1.0
            if q_norm == 0.0:
                q_norm = 1.0
            vector_scores = (np.dot(embeddings, q_emb) / (norms * q_norm)).tolist()
        else:
            vector_scores = [_cosine(embedding, record.embedding) for record in records_to_score]

        for idx, record in enumerate(records_to_score):
            vector_score = vector_scores[idx]
            keyword_score = len(keyword_set.intersection(record.keywords)) * 0.1
            tag_score = len(keyword_set.intersection(record.tags)) * 0.05
            score = vector_score + keyword_score + tag_score
            if score > 0.2:
                scored.append((score, record.memory_id))
        scored.sort(reverse=True)
        return [memory_id for _, memory_id in scored[:5]]

    def _upsert_links(self, source_id: str, target_ids: list[str]) -> None:
        for target_id in target_ids:
            if self.db_type == "postgres":
                with self._conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO memory_links (source_memory_id, target_memory_id, created_at)
                        VALUES (%s, %s, %s) ON CONFLICT DO NOTHING
                        """,
                        (source_id, target_id, time.time()),
                    )
            else:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_links (source_memory_id, target_memory_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (source_id, target_id, time.time()),
                )
                self._conn.commit()

    def _mark_accessed(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        if self.db_type == "postgres":
            with self._conn.cursor() as cur:
                cur.executemany(
                    "UPDATE memories SET access_count = access_count + 1 WHERE memory_id = %s",
                    [(memory_id,) for memory_id in memory_ids],
                )
        else:
            self._conn.executemany(
                "UPDATE memories SET access_count = access_count + 1 WHERE memory_id = ?",
                [(memory_id,) for memory_id in memory_ids],
            )
            self._conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        if self.db_type == "postgres":
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = '{table}' AND column_name = '{column}'
                    """
                )
                if not cur.fetchone():
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        else:
            columns = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()}
            if column not in columns:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _row_to_record(row: sqlite3.Row | dict) -> MemoryRecord:
    return MemoryRecord(
        memory_id=str(row["memory_id"]),
        source_agent=str(row["source_agent"]),
        created_at=float(row["created_at"]),
        task_topic=str(row["task_topic"]),
        summary=str(row["summary"]),
        tags=list(json.loads(row["tags"])),
        embedding=[float(value) for value in json.loads(row["embedding"])],
        keywords=list(json.loads(row["keywords"])) if "keywords" in row.keys() else [],
        links=list(json.loads(row["links"])) if "links" in row.keys() else [],
        access_count=int(row["access_count"]) if "access_count" in row.keys() else 0,
    )


def _terms(text: str) -> set[str]:
    terms = set()
    for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
        if len(token) > 1 or "\u4e00" <= token <= "\u9fff":
            terms.add(token)
    return terms


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
