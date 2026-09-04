#!/usr/bin/env python3
"""CPU contracts for Teacher-v9.2 hotspot/trust response."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import torch

from relational_teacher_v92_hotspot_trust_contract import (
    CANDIDATES,
    rank_candidates,
    response_checks,
)
from relational_teacher_v92_hotspot_trust_objective import (
    hotspot_listwise_macro_loss,
    hotspot_margin_macro_loss,
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
        "hotspot_margin_macro": 0.4,
        "hotspot_listwise_macro": 2.0,
        "background_trust": 0.0,
        "prompt_invariance": 0.01,
        "negative_mean": 0.02,
        "negative_max": 0.20,
        "per_prompt_metrics": [copy.deepcopy(prompt), copy.deepcopy(prompt)],
    }


def main() -> None:
    targets = torch.zeros((1, 3, 24, 6), dtype=torch.float32)
    masks = torch.zeros((1, 3, 24), dtype=torch.bool)
    bad = torch.zeros((1, 24, 6), dtype=torch.float32)
    good = torch.zeros_like(bad)
    for slot in range(3):
        start = slot * 8
        masks[0, slot, start : start + 8] = True
        targets[0, slot, start, :] = 1.0
        targets[0, slot, start + 1, :] = 0.8
        good[0, start, :] = 1.0
        good[0, start + 1, :] = 0.8
    bad_margin, bad_margin_rows = hotspot_margin_macro_loss(bad, targets, masks)
    good_margin, good_margin_rows = hotspot_margin_macro_loss(good, targets, masks)
    bad_listwise, bad_listwise_rows = hotspot_listwise_macro_loss(bad, targets, masks)
    good_listwise, good_listwise_rows = hotspot_listwise_macro_loss(good, targets, masks)
    if not float(good_margin) < float(bad_margin):
        raise AssertionError("hotspot margin does not prefer GT hotspot ordering")
    if not float(good_listwise) < float(bad_listwise):
        raise AssertionError("listwise loss does not prefer GT hotspot ordering")
    if bad_margin_rows.shape != (1, 3) or bad_listwise_rows.shape != (1, 3):
        raise AssertionError("hotspot losses are not equal three-object macros")

    before = _panel()
    after = _panel(
        topk=(0.21, 0.27, 0.17),
        recall=(0.599, 0.699, 0.499),
        mae=(0.301, 0.201, 0.401),
    )
    after["hotspot_margin_macro"] = 0.35
    after["hotspot_listwise_macro"] = 1.9
    after["background_trust"] = 0.001
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
        raise AssertionError("valid hotspot/trust response was rejected")
    missing_high = copy.deepcopy(after)
    for prompt in missing_high["per_prompt_metrics"]:
        prompt["instances"]["chair_06"]["topk_overlap"] = 0.15
    failed = response_checks(
        before=before,
        after=missing_high,
        base_v5_dense=(0.05, 0.04, 0.06),
        candidate_v5_dense=(0.0501, 0.0401, 0.0601),
    )
    if failed["high_chair_mean_topk_improves"] is not False:
        raise AssertionError("missing High-Chair hotspot gain was accepted")

    # Unlike v9.1, candidate objective ratios must not be scalar copies.  This
    # is what makes a clipped-gradient grid scientifically meaningful.
    keys = (
        "active_support_weight",
        "hotspot_margin_weight",
        "hotspot_listwise_weight",
        "absolute_negative_weight",
        "background_trust_weight",
        "prompt_weight",
        "v5_preservation_weight",
    )
    normalized = {
        tuple(round(float(row[key]) / float(row["hotspot_margin_weight"]), 8) for key in keys)
        for row in CANDIDATES
    }
    if len(normalized) != len(CANDIDATES):
        raise AssertionError("hotspot/trust candidates collapse to scalar copies")

    rows = []
    for index, candidate in enumerate(CANDIDATES):
        row_before = _panel()
        row_after = _panel(topk=(0.21, 0.26, 0.16))
        # Candidate 2 has the best worst-object and total top-k gain.
        gain = (0.01, 0.02, 0.015)[index]
        for prompt in row_after["per_prompt_metrics"]:
            prompt["instances"]["bed_01"]["topk_overlap"] = 0.20 + gain
            prompt["instances"]["chair_01"]["topk_overlap"] = 0.25 + gain
            prompt["instances"]["chair_06"]["topk_overlap"] = 0.15 + gain
        row_after["background_trust"] = 0.001 + index * 0.0001
        rows.append({**candidate, "eligible": True, "before": row_before, "after": row_after})
    if rank_candidates(rows)[0] != CANDIDATES[1]["name"]:
        raise AssertionError("hotspot/trust response ranking changed")

    source = Path(__file__).with_name(
        "preflight_relational_teacher_v92_hotspot_trust.py"
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
        raise AssertionError("response grid must load exactly one v9 scene bundle")
    if "records[TRAIN_SCENE]" not in source:
        raise AssertionError("response grid is not statically bound to room_0101")
    if "torch.save" in source or "load_ckpt(model, str(corrected" in source:
        raise AssertionError("response grid may not save/load failed candidate state")
    if '"authorizes_overfit120": False' not in source:
        raise AssertionError("response grid accidentally authorizes 120-step training")

    print("[PASS] Teacher-v9.2 hotspot/trust CPU contract")
    print("[PASS] conservative hotspot margin and within-object listwise ordering")
    print("[PASS] per-object top-k, trust, ratio-grid and room_0101-only guards")


if __name__ == "__main__":
    main()
