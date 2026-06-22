from __future__ import annotations

import hashlib
import atexit
import array
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
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
        self.backend = os.getenv("STATE_BACKEND", "shared_memory").lower()
        self._shared_segments: list[shared_memory.SharedMemory] = []
        self._closed = False
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        atexit.register(self.close)

    def put(
        self,
        producer_agent: str,
        task_id: str,
        state_type: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> StateRecord:
        payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        state_id = _state_id(producer_agent, task_id, state_type, payload_text)
        record_metadata = dict(metadata or {})
        stored_payload: Any = json.loads(payload_text)
        size_bytes = len(payload_text.encode("utf-8"))

        if self.backend in {"shared_memory", "shm"} and state_type == "embedding" and _is_float_list(payload):
            shared_state = self._put_embedding_shared_memory(state_id, payload)
            if shared_state is not None:
                stored_payload = shared_state["payload"]
                record_metadata.update(shared_state["metadata"])
                size_bytes = int(shared_state["metadata"]["shm_size"])
        else:
            record_metadata.setdefault("storage_backend", "sqlite")

        record = StateRecord(
            state_id=state_id,
            producer_agent=producer_agent,
            task_id=task_id,
            state_type=state_type,
            created_at=time.time(),
            size_bytes=size_bytes,
            metadata=record_metadata,
            payload=stored_payload,
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
                json.dumps(record.payload, ensure_ascii=False, sort_keys=True, default=str),
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
        if not row:
            return None
        record = _row_to_record(row)
        if record.metadata.get("storage_backend") == "shared_memory" and os.getenv("STATE_READ_PAYLOAD", "0") == "1":
            vector = self.read_shared_vector(record)
            if vector is not None:
                record.payload = vector
        return record

    def read_shared_vector(self, record: StateRecord) -> list[float] | None:
        """Read a shared-memory vector by reference.

        The default reports keep only metadata. This method demonstrates the
        receiver-side path: attach to the named segment and read float64 values
        directly from the shared memory buffer.
        """

        shm_name = record.metadata.get("shm_name")
        length = int(record.metadata.get("shape", [0])[0] or 0)
        if not shm_name or not length:
            return None
        shm = None
        try:
            shm = shared_memory.SharedMemory(name=str(shm_name), create=False)
            vector = array.array("d")
            vector.frombytes(bytes(shm.buf[: length * 8]))
            return [float(value) for value in vector]
        except FileNotFoundError:
            return None
        finally:
            if shm is not None:
                shm.close()

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
        if self._closed:
            return
        for segment in self._shared_segments:
            try:
                segment.close()
                segment.unlink()
            except FileNotFoundError:
                pass
        self._shared_segments.clear()
        self._conn.close()
        self._closed = True

    def _put_embedding_shared_memory(self, state_id: str, payload: Any) -> dict[str, Any] | None:
        try:
            vector = array.array("d", [float(value) for value in payload])
            shm = shared_memory.SharedMemory(create=True, size=len(vector) * 8)
            shm.buf[: len(vector) * 8] = vector.tobytes()
            self._shared_segments.append(shm)
            return {
                "metadata": {
                    "storage_backend": "shared_memory",
                    "shm_name": shm.name,
                    "shm_size": len(vector) * 8,
                    "dtype": "float64",
                    "shape": [len(vector)],
                    "zero_copy_receiver": True,
                    "consumer_hint": "numpy.ndarray(shape, dtype='float64', buffer=SharedMemory(name).buf)",
                },
                "payload": {
                    "state_ref": state_id,
                    "storage_backend": "shared_memory",
                    "preview": [round(float(value), 6) for value in vector[:8]],
                    "length": len(vector),
                },
            }
        except Exception:
            return None

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


def _is_float_list(payload: Any) -> bool:
    if not isinstance(payload, list) or not payload:
        return False
    return all(isinstance(value, (int, float)) for value in payload)
