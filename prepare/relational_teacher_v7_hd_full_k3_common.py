#!/usr/bin/env python3
"""Metrics and seeds for Teacher-v7 full train-only K=3."""

from __future__ import annotations

import numpy as np

from relational_teacher_v6_contract import PROMPTS, SITTABLE_CATEGORY_IDS
from relational_teacher_v7_hd_k3_common import generation_seeds
from relational_teacher_v7_hd_rollout_common import semantic_metrics, target_mae


def aggregate_full(
    relational_gt, relational_base, relational_candidate, instance_ids,
    category_ids, target_instance_ids, scene_ids, target_names, strata,
    motion_ids, prompt_ids, v5_gt, v5_base, v5_candidate, v5_ids, v5_targets,
):
    cases = []
    for index in range(len(relational_gt)):
        mask = instance_ids[index] == int(target_instance_ids[index])
        cases.append({
            "index": index,
            "scene_id": str(scene_ids[index]),
            "target_name": str(target_names[index]),
            "training_stratum": str(strata[index]),
            "motion_id": str(motion_ids[index]),
            "prompt_id": str(prompt_ids[index]),
            "target_mae": {
                "base": target_mae(relational_base[index], relational_gt[index], mask),
                "candidate": target_mae(relational_candidate[index], relational_gt[index], mask),
            },
            "semantic": {
                "base": semantic_metrics(relational_base[index], instance_ids[index], category_ids[index]),
                "candidate": semantic_metrics(relational_candidate[index], instance_ids[index], category_ids[index]),
            },
        })

    pair_rows = []
    pair_keys = sorted(set(
        (str(scene_ids[index]), str(strata[index]), str(motion_ids[index]))
        for index in range(len(scene_ids))
    ))
    for scene_id, stratum, motion_id in pair_keys:
        indices = [index for index in range(len(scene_ids))
                   if str(scene_ids[index]) == scene_id
                   and str(strata[index]) == stratum
                   and str(motion_ids[index]) == motion_id]
        by_prompt = {str(prompt_ids[index]): index for index in indices}
        if set(by_prompt) != set(PROMPTS):
            raise ValueError("full train prompt pair is incomplete")
        watch, write = by_prompt["sit_watch_v1"], by_prompt["sit_write_v1"]
        candidate_mask = np.isin(category_ids[watch], list(SITTABLE_CATEGORY_IDS))
        pair_rows.append({
            "scene_id": scene_id,
            "training_stratum": stratum,
            "motion_id": motion_id,
            "base_mse": float(np.square(
                relational_base[watch, candidate_mask]
                - relational_base[write, candidate_mask]).mean()),
            "candidate_mse": float(np.square(
                relational_candidate[watch, candidate_mask]
                - relational_candidate[write, candidate_mask]).mean()),
        })

    replay_rows = []
    for index, (sample_id, target) in enumerate(zip(v5_ids, v5_targets)):
        base_mae = float(np.abs(v5_base[index] - v5_gt[index]).mean())
        candidate_mae = float(np.abs(v5_candidate[index] - v5_gt[index]).mean())
        replay_rows.append({
            "sample_id": str(sample_id),
            "target": str(target),
            "base_mae": base_mae,
            "candidate_mae": candidate_mae,
            "relative_degradation": (candidate_mae - base_mae) / max(base_mae, 1e-12),
        })
    return compose_metrics(cases, pair_rows, replay_rows)


def target_summary(rows):
    base = float(np.mean([row["target_mae"]["base"] for row in rows]))
    candidate = float(np.mean([row["target_mae"]["candidate"] for row in rows]))
    return {
        "count": len(rows),
        "target_mae_base": base,
        "target_mae_candidate": candidate,
        "target_mae_relative_change": (candidate - base) / max(base, 1e-12),
        "case_win_rate": float(np.mean([
            row["target_mae"]["candidate"] < row["target_mae"]["base"]
            for row in rows
        ])),
    }


def compose_metrics(cases, pair_rows, replay_rows):
    high_desk = [row for row in cases
                 if row["training_stratum"] == "high_desk_chair_06"]
    target_groups = {}
    for key in sorted(set(
        f"{row['scene_id']}|{row['training_stratum']}" for row in cases
    )):
        rows = [row for row in cases
                if f"{row['scene_id']}|{row['training_stratum']}" == key]
        summary = target_summary(rows)
        target_groups[key] = {
            "count": summary["count"],
            "base_mae": summary["target_mae_base"],
            "candidate_mae": summary["target_mae_candidate"],
            "relative_change": summary["target_mae_relative_change"],
            "case_win_rate": summary["case_win_rate"],
        }
    v5_groups = {}
    for target in sorted(set(row["target"] for row in replay_rows)):
        rows = [row for row in replay_rows if row["target"] == target]
        base = float(np.mean([row["base_mae"] for row in rows]))
        candidate = float(np.mean([row["candidate_mae"] for row in rows]))
        v5_groups[target] = {
            "count": len(rows),
            "base_mae": base,
            "candidate_mae": candidate,
            "relative_degradation": (candidate - base) / max(base, 1e-12),
        }
    overall = target_summary(cases)
    overall.update({
        "semantic_violation_base": float(np.mean([
            row["semantic"]["base"]["semantic_violation"] for row in cases
        ])),
        "semantic_violation_candidate": float(np.mean([
            row["semantic"]["candidate"]["semantic_violation"] for row in cases
        ])),
        "prompt_invariance_mse_base": float(np.mean([row["base_mse"] for row in pair_rows])),
        "prompt_invariance_mse_candidate": float(np.mean([row["candidate_mse"] for row in pair_rows])),
        "v5_replay_mae_base": float(np.mean([row["base_mae"] for row in replay_rows])),
        "v5_replay_mae_candidate": float(np.mean([row["candidate_mae"] for row in replay_rows])),
    })
    overall["v5_replay_relative_degradation"] = (
        overall["v5_replay_mae_candidate"] - overall["v5_replay_mae_base"]
    ) / max(overall["v5_replay_mae_base"], 1e-12)
    return {
        "relational_cases": cases,
        "prompt_pairs": pair_rows,
        "v5_replay_cases": replay_rows,
        "target_groups": target_groups,
        "v5_target_groups": v5_groups,
        "high_desk": target_summary(high_desk),
        "overall": overall,
    }


def per_generation_metrics(reference, relational_base, relational_candidate,
                           v5_base, v5_candidate):
    metadata = {
        key: [str(value) for value in reference[key].tolist()]
        for key in ("rel_scene_ids", "rel_target_names", "rel_strata",
                    "rel_motion_ids", "rel_prompt_ids", "v5_ids", "v5_targets")
    }
    return [aggregate_full(
        reference["rel_gt"], relational_base[:, generation],
        relational_candidate[:, generation], reference["rel_instance_ids"],
        reference["rel_category_ids"], reference["rel_target_instance_ids"],
        metadata["rel_scene_ids"], metadata["rel_target_names"],
        metadata["rel_strata"], metadata["rel_motion_ids"],
        metadata["rel_prompt_ids"], reference["v5_gt"], v5_base[:, generation],
        v5_candidate[:, generation], metadata["v5_ids"], metadata["v5_targets"],
    ) for generation in range(3)]


def pooled_metrics(generations):
    cases, pairs, replay = [], [], []
    for generation, metrics in enumerate(generations):
        cases.extend({**row, "generation": generation}
                     for row in metrics["relational_cases"])
        pairs.extend({**row, "generation": generation}
                     for row in metrics["prompt_pairs"])
        replay.extend({**row, "generation": generation}
                      for row in metrics["v5_replay_cases"])
    return compose_metrics(cases, pairs, replay)


__all__ = [
    "aggregate_full", "generation_seeds", "per_generation_metrics",
    "pooled_metrics",
]
