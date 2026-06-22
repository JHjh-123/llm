from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class StateRecord:
    state_id: str
    producer_agent: str
    task_id: str
    state_type: str
    created_at: float
    size_bytes: int
    metadata: dict[str, Any]
    payload: Any


class StateStore:
    """Persistent exchange area for non-text intermediate state.

    Agents pass state IDs in protocol messages while vectors, tool results, and
    other bulky intermediate objects stay in this local store.
    """

    def __init__(self, db_path: str | Path | None = None, reset: bool = False) -> None:
        self.db_path = Path(db_path or os.getenv("STATE_PATH", "data/state.sqlite"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.db_path.exists():
            self.db_path.unlink()
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def put(
        self,
        producer_agent: str,
        task_id: str,
        state_type: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> StateRecord:
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        size_bytes = len(payload_text.encode("utf-8"))
        state_id = _state_id(producer_agent, task_id, state_type, payload_text)
        record = StateRecord(
            state_id=state_id,
            producer_agent=producer_agent,
            task_id=task_id,
            state_type=state_type,
            created_at=time.time(),
            size_bytes=size_bytes,
            metadata=metadata or {},
            payload=json.loads(payload_text),
        )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO states
            (state_id, producer_agent, task_id, state_type, created_at, size_bytes, metadata, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.state_id,
                record.producer_agent,
                record.task_id,
                record.state_type,
                record.created_at,
                record.size_bytes,
                json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                payload_text,
            ),
        )
        self._conn.commit()
        return record

    def get(self, state_id: str) -> StateRecord | None:
        row = self._conn.execute(
            """
            SELECT state_id, producer_agent, task_id, state_type, created_at, size_bytes, metadata, payload
            FROM states
            WHERE state_id = ?
            """,
            (state_id,),
        ).fetchone()
        return _row_to_record(row) if row else None

    def list_for_task(self, task_id: str) -> list[StateRecord]:
        rows = self._conn.execute(
            """
            SELECT state_id, producer_agent, task_id, state_type, created_at, size_bytes, metadata, payload
            FROM states
            WHERE task_id = ?
            ORDER BY created_at ASC
            """,
            (task_id,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def to_dict(self, include_payload: bool = False) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT state_id, producer_agent, task_id, state_type, created_at, size_bytes, metadata, payload
            FROM states
            ORDER BY created_at ASC
            """
        ).fetchall()
        records = []
        for row in rows:
            record = asdict(_row_to_record(row))
            if not include_payload:
                record.pop("payload", None)
            records.append(record)
        return records

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS states (
                state_id TEXT PRIMARY KEY,
                producer_agent TEXT NOT NULL,
                task_id TEXT NOT NULL,
                state_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                size_bytes INTEGER NOT NULL,
                metadata TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_states_task ON states(task_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_states_type ON states(state_type)")
        self._conn.commit()


def _row_to_record(row: sqlite3.Row) -> StateRecord:
    return StateRecord(
        state_id=str(row["state_id"]),
        producer_agent=str(row["producer_agent"]),
        task_id=str(row["task_id"]),
        state_type=str(row["state_type"]),
        created_at=float(row["created_at"]),
        size_bytes=int(row["size_bytes"]),
        metadata=dict(json.loads(row["metadata"])),
        payload=json.loads(row["payload"]),
    )


def _state_id(producer_agent: str, task_id: str, state_type: str, payload_text: str) -> str:
    digest = hashlib.sha1(f"{producer_agent}:{task_id}:{state_type}:{payload_text}".encode("utf-8")).hexdigest()
    return f"state_{digest[:12]}"
