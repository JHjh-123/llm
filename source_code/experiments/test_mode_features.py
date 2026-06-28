from __future__ import annotations

import os

from multi_agent.runner import _mode_feature_enabled


def main() -> None:
    previous = os.environ.get("FEATURE_STATE_EXCHANGE")
    try:
        os.environ.pop("FEATURE_STATE_EXCHANGE", None)
        assert not _mode_feature_enabled(
            "FEATURE_STATE_EXCHANGE",
            default=True,
            mode="text",
            structured_only=True,
        )
        assert _mode_feature_enabled(
            "FEATURE_STATE_EXCHANGE",
            default=True,
            mode="structured",
            structured_only=True,
        )

        os.environ["FEATURE_STATE_EXCHANGE"] = "0"
        assert not _mode_feature_enabled(
            "FEATURE_STATE_EXCHANGE",
            default=True,
            mode="structured",
            structured_only=True,
        )

        os.environ["FEATURE_STATE_EXCHANGE"] = "1"
        assert not _mode_feature_enabled(
            "FEATURE_STATE_EXCHANGE",
            default=True,
            mode="text",
            structured_only=True,
        )
    finally:
        if previous is None:
            os.environ.pop("FEATURE_STATE_EXCHANGE", None)
        else:
            os.environ["FEATURE_STATE_EXCHANGE"] = previous

    print("mode feature smoke test passed")


if __name__ == "__main__":
    main()
