#!/usr/bin/env python3
"""Contract for the v9.2 CPU/CUDA comparison correction."""

from __future__ import annotations

from validate_relational_teacher_v92_hotspot_trust_v2 import _reduction_close


def main() -> None:
    _reduction_close(
        {"listwise": [4.123456, 5.000000]},
        {"listwise": [4.123464, 5.000009]},
        "accepted reduction",
    )
    rejected = False
    try:
        _reduction_close({"listwise": [4.0]}, {"listwise": [4.001]}, "tamper")
    except ValueError:
        rejected = True
    if not rejected:
        raise AssertionError("validator correction accepted material numeric tampering")
    print("[PASS] Teacher-v9.2 validator CPU/CUDA tolerance contract")
    print("[PASS] reduction noise accepted; material value change rejected")


if __name__ == "__main__":
    main()

