from __future__ import annotations

from multi_agent.protocol import Message, validate_structured_payload


def main() -> None:
    valid_payload = {
        "a": "test",
        "in": {"value": "input"},
        "out": "ok",
        "refs": [],
        "state": {"kind": "smoke"},
    }
    validate_structured_payload(valid_payload)
    message = Message.structured(
        sender="tester",
        receiver="runtime",
        payload=valid_payload,
        task_id="task_protocol_smoke",
    )
    wire = message.to_wire()
    assert '"a":"test"' in wire

    try:
        validate_structured_payload({"a": "bad", "out": "missing refs"})
    except ValueError as exc:
        assert "refs" in str(exc)
    else:
        raise AssertionError("invalid payload unexpectedly passed validation")

    print("protocol smoke test passed")


if __name__ == "__main__":
    main()
