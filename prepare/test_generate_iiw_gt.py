#!/usr/bin/env python3
"""Deterministic unit tests for ``generate_iiw_gt.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from generate_iiw_gt import (  # noqa: E402
    CMDM_PROXY_FROM_NATIVE,
    NATIVE_BODY_JOINT_GROUPS,
    compute_phase_iiw,
    distance_kernel,
    phase_ranges,
    validate_prefix_mask,
)


def test_unity_spine2_mapping() -> None:
    assert NATIVE_BODY_JOINT_GROUPS["spine"] == (9,)
    print("[PASS] Unity YBot Spine2 maps to HumanML22 index 9")


def test_distance_kernel() -> None:
    lower = 0.1
    upper = 0.55
    distance = np.asarray(
        [0.0, lower, 0.325, upper, upper + 1e-6], dtype=np.float32
    )
    actual = distance_kernel(distance, lower, upper)
    expected = np.asarray(
        [
            1.0,
            (lower + upper) / (lower + upper),
            (lower + upper) / (0.325 + upper),
            (lower + upper) / (upper + upper),
            0.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    print("[PASS] exact InterFaceRays distance kernel")


def test_phase_ranges() -> None:
    ranges = phase_ranges(118, 8)
    counts = np.asarray([end - start for start, end in ranges])
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 118
    assert int(counts.sum()) == 118
    assert np.all(counts > 0)
    print("[PASS] phase ranges cover valid frames exactly")


def test_prefix_mask() -> None:
    valid = 11
    mask = np.ones((196,), dtype=bool)
    mask[:valid] = False
    validate_prefix_mask(mask, valid)
    invalid = mask.copy()
    invalid[5] = True
    invalid[valid] = False
    try:
        validate_prefix_mask(invalid, valid)
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous motion mask was accepted")
    try:
        validate_prefix_mask(np.ones((20,), dtype=bool), 11)
    except ValueError:
        pass
    else:
        raise AssertionError("non-production CMDM horizon was accepted")
    print("[PASS] padded frames cannot enter IIW targets")


def make_synthetic_motion(num_frames: int = 16) -> np.ndarray:
    motion = np.zeros((num_frames, 22, 3), dtype=np.float32)
    # Separate all joints slightly so every native body group has a distinct
    # but finite signal.  Move along X to exercise phase ordering.
    for frame in range(num_frames):
        motion[frame, :, 0] = np.linspace(-0.2, 0.2, 22)
        motion[frame, :, 1] = -0.4 + frame * 0.05
        motion[frame, :, 2] = np.linspace(0.0, 1.0, 22)
    return motion


def test_chunking_and_channel_proxy() -> None:
    rng = np.random.default_rng(20260812)
    motion = make_synthetic_motion()
    scene = rng.uniform(-1.0, 1.0, size=(37, 3)).astype(np.float32)
    small = compute_phase_iiw(
        motion, scene, 8, 0.1, 0.55, point_chunk_size=5
    )
    full = compute_phase_iiw(
        motion, scene, 8, 0.1, 0.55, point_chunk_size=len(scene)
    )
    for small_value, full_value in zip(small[:3], full[:3]):
        np.testing.assert_allclose(
            small_value, full_value, rtol=0.0, atol=1e-6
        )
    native = small[0]
    proxy = native[..., CMDM_PROXY_FROM_NATIVE]
    for output_index, native_index in enumerate(CMDM_PROXY_FROM_NATIVE):
        np.testing.assert_array_equal(
            proxy[..., output_index], native[..., native_index]
        )
    assert np.isfinite(native).all()
    assert np.all((native >= 0.0) & (native <= 1.0))
    print("[PASS] chunked IIW equals full computation")
    print("[PASS] native-to-CMDM proxy channel order")


def test_time_reversal() -> None:
    motion = make_synthetic_motion()
    scene = np.asarray(
        [[0.0, -0.4, 0.0], [0.0, 0.3, 0.5]], dtype=np.float32
    )
    forward = compute_phase_iiw(
        motion, scene, 8, 0.1, 0.55, point_chunk_size=2
    )[0]
    reverse = compute_phase_iiw(
        motion[::-1].copy(), scene, 8, 0.1, 0.55, point_chunk_size=2
    )[0]
    np.testing.assert_allclose(
        forward[::-1], reverse, rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        forward.max(axis=0), reverse.max(axis=0), rtol=0.0, atol=1e-6
    )
    if np.allclose(forward, reverse, rtol=0.0, atol=1e-6):
        raise AssertionError("phase targets unexpectedly lost temporal order")
    print("[PASS] phase IIW preserves temporal order")


def main() -> None:
    test_unity_spine2_mapping()
    test_distance_kernel()
    test_phase_ranges()
    test_prefix_mask()
    test_chunking_and_channel_proxy()
    test_time_reversal()
    print("[PASS] generate_iiw_gt unit contract")


if __name__ == "__main__":
    main()
