#!/usr/bin/env python3
"""CPU contracts for Teacher-v9.3 evaluation-aligned exact top-k."""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path

import numpy as np
import torch

from relational_teacher_v93_exact_topk_contract import (
    CANDIDATES,
    TRAINING_TIMESTEPS,
    rank_candidates,
    response_checks,
)
from relational_teacher_v93_exact_topk_objective import (
    ACTIVE_THRESHOLD,
    TOP_FRACTION,
    evaluation_target_topk,
    exact_topk_swap_macro_loss,
)


def _panel(topk=(0.20, 0.25, 0.15), recall=(0.6, 0.7, 0.5), mae=(0.3, 0.2, 0.4)):
    names = ("bed_01", "chair_01", "chair_06")
    prompt = {
        "instances": {
            name: {
                "topk_overlap": float(topk[index]),
                "soft_recall": float(recall[index]),
                "active_support_mae": float(mae[index]),
            }
            for index, name in enumerate(names)
        }
    }
    return {
        "exact_topk_swap_macro": 0.2,
        "per_instance_exact_topk_swap": [0.2, 0.2, 0.2],
        "per_instance_exact_topk_overlap": list(topk),
        "background_trust": 0.0,
        "prompt_invariance": 0.01,
        "negative_mean": 0.02,
        "negative_max": 0.20,
        "per_prompt_metrics": [copy.deepcopy(prompt), copy.deepcopy(prompt)],
    }


def main() -> None:
    # Verify target-set construction exactly matches the evaluation formula:
    # threshold first, then ceil(active_count * 0.25), then value ranking.
    target = torch.tensor([0.9, 0.8, 0.7, 0.31, 0.29, 0.2, 0.1, 0.0])
    mask = torch.ones(8, dtype=torch.bool)
    selected, object_indices = evaluation_target_topk(target, mask)
    active = np.flatnonzero(target.numpy() >= ACTIVE_THRESHOLD)
    count = max(1, int(math.ceil(active.size * TOP_FRACTION)))
    expected = active[np.lexsort((active, -target.numpy()[active]))[:count]]
    if not np.array_equal(selected.numpy(), expected) or object_indices.numel() != 8:
        raise AssertionError("training and evaluation target top-k definitions differ")
    if selected.numel() == int(math.ceil(mask.sum().item() * TOP_FRACTION)):
        raise AssertionError("exact top-k regressed to whole-object fraction")

    targets = torch.zeros((1, 3, 24, 6), dtype=torch.float32)
    masks = torch.zeros((1, 3, 24), dtype=torch.bool)
    wrong = torch.zeros((1, 24, 6), dtype=torch.float32)
    correct = torch.zeros_like(wrong)
    for slot in range(3):
        start = slot * 8
        masks[0, slot, start : start + 8] = True
        targets[0, slot, start, :] = 0.9
        targets[0, slot, start + 1, :] = 0.8
        wrong[0, start + 5, :] = 0.95
        correct[0, start, :] = 0.95
    wrong_loss, wrong_rows, wrong_overlap = exact_topk_swap_macro_loss(
        wrong, targets, masks
    )
    correct_loss, correct_rows, correct_overlap = exact_topk_swap_macro_loss(
        correct, targets, masks
    )
    if not float(correct_loss) < float(wrong_loss):
        raise AssertionError("exact swap loss does not prefer the metric's GT top-k")
    if wrong_rows.shape != (1, 3) or wrong_overlap.shape != (1, 3):
        raise AssertionError("exact swap loss is not an equal three-object macro")
    if not torch.all(correct_overlap > wrong_overlap):
        raise AssertionError("exact swap overlap does not track all three objects")

    before = _panel()
    after = _panel(
        topk=(0.21, 0.27, 0.17),
        recall=(0.599, 0.699, 0.499),
        mae=(0.301, 0.201, 0.401),
    )
    after["per_instance_exact_topk_swap"] = [0.19, 0.18, 0.17]
    after["per_instance_exact_topk_overlap"] = [0.21, 0.27, 0.17]
    after["background_trust"] = 1e-6
    after["prompt_invariance"] = 0.0101
    after["negative_mean"] = 0.021
    after["negative_max"] = 0.205
    checks = response_checks(
        before=before,
        after=after,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if not all(checks.values()):
        raise AssertionError("valid exact top-k response was rejected")
    one_prompt_worse = copy.deepcopy(after)
    one_prompt_worse["per_prompt_metrics"][0]["instances"]["chair_06"]["topk_overlap"] = 0.14
    failed = response_checks(
        before=before,
        after=one_prompt_worse,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if failed["high_chair_each_prompt_topk_not_worse"] is not False:
        raise AssertionError("one-prompt High-Chair regression was accepted")

    rows = []
    for index, candidate in enumerate(CANDIDATES):
        candidate_after = copy.deepcopy(after)
        gain = (0.01, 0.03, 0.02)[index]
        for prompt in candidate_after["per_prompt_metrics"]:
            for name, base_value in (("bed_01", 0.20), ("chair_01", 0.25), ("chair_06", 0.15)):
                prompt["instances"][name]["topk_overlap"] = base_value + gain
        rows.append({**candidate, "eligible": True, "before": before, "after": candidate_after})
    if rank_candidates(rows)[0] != "exact_swap_lr020":
        raise AssertionError("exact top-k LR selection ranking changed")
    if [row["learning_rate"] for row in CANDIDATES] != [1e-5, 2e-5, 4e-5]:
        raise AssertionError("exact top-k learning-rate grid changed")

    source = Path(__file__).with_name(
        "preflight_relational_teacher_v93_exact_topk.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_train_scene_bundle"
    ]
    if len(loads) != 1 or "records[TRAIN_SCENE]" not in source:
        raise AssertionError("exact top-k response is not room_0101-only")
    if "torch.save" in source or "load_ckpt(model, str(failed_v92" in source:
        raise AssertionError("exact top-k response may not save/load failed state")
    if "timestep_values = TRAINING_TIMESTEPS" not in source:
        raise AssertionError("training timestep grid is not contract-bound")

    print("[PASS] Teacher-v9.3 exact top-k CPU contract")
    print("[PASS] GT>=0.3 then top-25% exactly matches evaluation semantics")
    print("[PASS] three-object swap, per-prompt, LR-grid and held-out guards")


if __name__ == "__main__":
    main()
