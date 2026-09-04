#!/usr/bin/env python3
"""CPU contracts for the Teacher-v9.1 active-support response gate."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import torch

from relational_teacher_v91_active_support_objective import (
    active_support_macro_loss,
    within_object_ranking_loss,
)
from relational_teacher_v91_loss_response_contract import (
    CANDIDATES,
    rank_candidates,
    response_checks,
)


def main() -> None:
    prediction = torch.zeros((1, 12, 6), dtype=torch.float32)
    targets = torch.zeros((1, 3, 12, 6), dtype=torch.float32)
    masks = torch.zeros((1, 3, 12), dtype=torch.bool)
    for slot in range(3):
        start = slot * 4
        masks[0, slot, start : start + 4] = True
        targets[0, slot, start, :] = 0.9 - 0.1 * slot
        targets[0, slot, start + 1, :] = 0.6 - 0.1 * slot
    active, rows = active_support_macro_loss(prediction, targets, masks)
    expected = torch.stack(
        [targets[0, slot][targets[0, slot] >= 0.30].square().mean() for slot in range(3)]
    ).mean()
    if not torch.allclose(active, expected) or rows.shape != (1, 3):
        raise AssertionError("active-support loss is not an equal three-object macro")

    bad_rank, _ = within_object_ranking_loss(prediction, targets, masks)
    good = prediction.clone()
    for slot in range(3):
        good[0, slot * 4, :] = 1.0
    good_rank, _ = within_object_ranking_loss(good, targets, masks)
    if not float(good_rank) < float(bad_rank):
        raise AssertionError("ranking loss does not prefer GT hotspot points")

    before = {
        "active_support_macro": 0.30,
        "within_object_ranking": 0.20,
        "per_instance_active_support": [0.30, 0.30, 0.30],
        "prompt_invariance": 0.010,
        "negative_mean": 0.020,
    }
    after = {
        "active_support_macro": 0.25,
        "within_object_ranking": 0.15,
        "per_instance_active_support": [0.28, 0.24, 0.23],
        "prompt_invariance": 0.0101,
        "negative_mean": 0.021,
    }
    checks = response_checks(
        before=before,
        after=after,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if not all(checks.values()):
        raise AssertionError("valid corrected response was rejected")
    bad = copy.deepcopy(after)
    bad["per_instance_active_support"][2] = 0.31
    failed = response_checks(
        before=before,
        after=bad,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if failed["high_chair_active_support_decreases"] is not False:
        raise AssertionError("worsened High Chair support was accepted")

    grid_rows = []
    for index, candidate in enumerate(CANDIDATES):
        grid_rows.append(
            {
                **candidate,
                "eligible": index != 0,
                "after": {
                    "active_support_macro": 0.20 + 0.01 * index,
                    "within_object_ranking": 0.10,
                },
                "candidate_v5_dense": [0.05, 0.04, 0.06],
            }
        )
    if rank_candidates(grid_rows) != [
        "support4_rank050_preserve4",
        "support8_rank100_preserve8",
    ]:
        raise AssertionError("candidate ranking changed")

    source = Path(__file__).with_name(
        "preflight_relational_teacher_v91_loss_response.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(calls) != 1:
        raise AssertionError("response grid must load one scene bundle exactly once")
    argument = calls[0].args[-1]
    slice_value = argument.slice if isinstance(argument, ast.Subscript) else None
    if isinstance(slice_value, ast.Index):
        slice_value = slice_value.value
    if not (
        isinstance(argument, ast.Subscript)
        and isinstance(argument.value, ast.Name)
        and argument.value.id == "records"
        and isinstance(slice_value, ast.Name)
        and slice_value.id == "TRAIN_SCENE"
    ):
        raise AssertionError("response grid is not statically bound to room_0101")
    if "torch.save" in source or "load_ckpt(model, str(failed" in source:
        raise AssertionError("response grid may not save/load failed candidate state")

    print("[PASS] Teacher-v9.1 active-support/ranking CPU contract")
    print("[PASS] equal three-object foreground and hotspot ordering losses")
    print("[PASS] response retention, ranking and room_0101-only guards")


if __name__ == "__main__":
    main()
