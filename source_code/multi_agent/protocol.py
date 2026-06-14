from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


PROTOCOL_VERSION = "agent-msg/v1"
STRUCTURED_PAYLOAD_SCHEMA = {
    "required": {
        "a": "short action name",
        "out": "compact output text",
        "refs": "list of memory or state references",
    },
    "optional": {
        "in": "compact structured input summary",
        "state": "non-text state metadata",
        "version": "protocol version for negotiation messages",
        "cap": "agent capability list for handshake messages",
        "accept": "accepted communication modes for handshake messages",
        "fields": "protocol field mapping for negotiation messages",
    },
}


@dataclass
class Message:
    message_id: str
    task_id: str
    parent_id: str | None
    sender: str
    receiver: str
    mode: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    error: dict[str, Any] | None = None

    @classmethod
    def structured(
        cls,
        sender: str,
        receiver: str,
        payload: dict[str, Any],
        task_id: str,
        parent_id: str | None = None,
        error: dict[str, Any] | None = None,
    ) -> "Message":
        validate_structured_payload(payload)
        content = str(payload.get("out", ""))
        return cls(
            message_id=_new_id("msg"),
            task_id=task_id,
            parent_id=parent_id,
            sender=sender,
            receiver=receiver,
            mode="structured",
            content=content,
            payload=payload,
            error=error,
        )

    def to_wire(self) -> str:
        if self.mode == "structured":
            envelope = {
                "mid": self.message_id,
                "tid": self.task_id,
                "pid": self.parent_id,
                "ts": round(self.created_at, 6),
                "f": self.sender,
                "t": self.receiver,
                "s": PROTOCOL_VERSION,
                "p": self.payload,
            }
            if self.error:
                envelope["err"] = self.error
            return json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (
            f"MSG {self.message_id}\nTASK {self.task_id}\n"
            f"FROM {self.sender}\nTO {self.receiver}\n\n{self.content}"
        )


def make_text_message(
    sender: str,
    receiver: str,
    content: str,
    task_id: str,
    parent_id: str | None = None,
) -> Message:
    return Message(
        message_id=_new_id("msg"),
        task_id=task_id,
        parent_id=parent_id,
        sender=sender,
        receiver=receiver,
        mode="text",
        content=content,
        payload={},
    )


def make_handshake(task_id: str, agent_name: str, capabilities: list[str]) -> Message:
    return Message.structured(
        sender=agent_name,
        receiver="runtime",
        task_id=task_id,
        payload={
            "a": "handshake",
            "version": PROTOCOL_VERSION,
            "cap": capabilities,
            "accept": ["text", "structured"],
            "out": f"{agent_name} ready",
            "refs": [],
        },
    )


def make_protocol_mapping(task_id: str) -> Message:
    return Message.structured(
        sender="runtime",
        receiver="all",
        task_id=task_id,
        payload={
            "a": "protocol_map",
            "version": PROTOCOL_VERSION,
            "fields": {
                "a": "action",
                "in": "compact inputs",
                "out": "compact output",
                "refs": "memory or state references",
                "state": "non-text state metadata",
            },
            "schema": STRUCTURED_PAYLOAD_SCHEMA,
            "out": "protocol mapping negotiated",
            "refs": [],
        },
    )


def validate_structured_payload(payload: dict[str, Any]) -> None:
    missing = [field for field in STRUCTURED_PAYLOAD_SCHEMA["required"] if field not in payload]
    if missing:
        raise ValueError(f"Structured payload missing required fields: {', '.join(missing)}")
    if not isinstance(payload["a"], str) or not payload["a"]:
        raise ValueError("Structured payload field 'a' must be a non-empty string")
    if not isinstance(payload["out"], str):
        raise ValueError("Structured payload field 'out' must be a string")
    if not isinstance(payload["refs"], list):
        raise ValueError("Structured payload field 'refs' must be a list")
    if "state" in payload and not isinstance(payload["state"], dict):
        raise ValueError("Structured payload field 'state' must be a dict when present")
    if "cap" in payload and not isinstance(payload["cap"], list):
        raise ValueError("Structured payload field 'cap' must be a list when present")
    if "accept" in payload and not isinstance(payload["accept"], list):
        raise ValueError("Structured payload field 'accept' must be a list when present")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
