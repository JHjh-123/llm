from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field

from multi_agent.protocol import Message

TOKEN_COUNT_METHOD = os.getenv("TOKEN_COUNT_METHOD", "unicode_heuristic")


@dataclass
class MetricsSnapshot:
    mode: str
    task: str
    message_count: int = 0
    char_count: int = 0
    approx_token_count: int = 0
    non_text_transfer_count: int = 0
    non_text_transfer_size: int = 0
    memory_hit_count: int = 0
    memory_hit_ids: list[str] = field(default_factory=list)
    protocol_event_count: int = 0
    protocol_char_count: int = 0
    protocol_approx_token_count: int = 0
    token_count_method: str = TOKEN_COUNT_METHOD
    message_trace: list[dict[str, object]] = field(default_factory=list)
    non_text_transfer_trace: list[dict[str, object]] = field(default_factory=list)
    state_ref_ids: list[str] = field(default_factory=list)
    orchestrator: str = "sequential"
    elapsed_ms: float = 0.0


class MetricsCollector:
    def __init__(self, mode: str, task: str) -> None:
        self.snapshot = MetricsSnapshot(mode=mode, task=task)
        self._started_at = time.perf_counter()

    def record_message(self, message: Message) -> None:
        wire = message.to_wire()
        approx_tokens = _count_tokens(wire)
        self.snapshot.message_count += 1
        self.snapshot.char_count += len(wire)
        self.snapshot.approx_token_count += approx_tokens
        self.snapshot.message_trace.append(
            {
                "message_id": message.message_id,
                "parent_id": message.parent_id,
                "from": message.sender,
                "to": message.receiver,
                "mode": message.mode,
                "action": message.payload.get("a"),
                "refs": message.payload.get("refs", []),
                "state": message.payload.get("state", {}),
                "chars": len(wire),
                "approx_tokens": approx_tokens,
            }
        )

    def record_non_text_transfer(self, transfer_type: str, size: int, state_id: str | None = None) -> None:
        self.snapshot.non_text_transfer_count += 1
        self.snapshot.non_text_transfer_size += size
        event: dict[str, object] = {"transfer_type": transfer_type, "size": size}
        if state_id:
            event["state_id"] = state_id
            self.snapshot.state_ref_ids.append(state_id)
        self.snapshot.non_text_transfer_trace.append(event)

    def record_memory_hit(self, memory_id: str) -> None:
        self.snapshot.memory_hit_count += 1
        self.snapshot.memory_hit_ids.append(memory_id)

    def record_protocol_event(self, message: Message) -> None:
        wire = message.to_wire()
        self.snapshot.protocol_event_count += 1
        self.snapshot.protocol_char_count += len(wire)
        self.snapshot.protocol_approx_token_count += _count_tokens(wire)

    def set_orchestrator(self, name: str) -> None:
        self.snapshot.orchestrator = name

    def finish(self) -> dict[str, object]:
        self.snapshot.elapsed_ms = round((time.perf_counter() - self._started_at) * 1000, 3)
        return asdict(self.snapshot)


def _count_tokens(text: str) -> int:
    method = TOKEN_COUNT_METHOD.lower()
    if method == "char_approx_4":
        return max(1, len(text) // 4)
    if method == "whitespace":
        return max(1, len(text.split()))
    if method == "tiktoken":
        counted = _try_tiktoken(text)
        if counted is not None:
            return counted
    return _unicode_heuristic(text)


def _try_tiktoken(text: str) -> int | None:
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding(os.getenv("TIKTOKEN_ENCODING", "cl100k_base"))
        return len(encoding.encode(text))
    except Exception:
        return None


def _unicode_heuristic(text: str) -> int:
    tokens = 0
    ascii_buffer = []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            tokens += _flush_ascii(ascii_buffer)
            ascii_buffer.clear()
            tokens += 1
        elif char.isascii() and (char.isalnum() or char in "_-./"):
            ascii_buffer.append(char)
        elif char.isspace():
            tokens += _flush_ascii(ascii_buffer)
            ascii_buffer.clear()
        else:
            tokens += _flush_ascii(ascii_buffer)
            ascii_buffer.clear()
            tokens += 1
    tokens += _flush_ascii(ascii_buffer)
    return max(1, tokens)


def _flush_ascii(buffer: list[str]) -> int:
    if not buffer:
        return 0
    text = "".join(buffer)
    pieces = [piece for piece in re.split(r"[_\-./]+", text) if piece]
    return sum(max(1, (len(piece) + 3) // 4) for piece in pieces)
