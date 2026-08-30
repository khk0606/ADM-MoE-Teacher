#!/usr/bin/env python3
"""Metric composition and deterministic seeds for Teacher-v7 K=3."""

from __future__ import annotations

from relational_teacher_v7_hd_k3_contract import GENERATIONS
from relational_teacher_v7_hd_rollout_common import aggregate_metrics, stable_seeds


def generation_seeds(seed: int, domain: str, case_id: str, generation: int):
    if generation == 0:
        return stable_seeds(seed, domain, case_id)
    return stable_seeds(seed, domain + "_k3", case_id + f"|g={generation}")


def per_generation_metrics(reference, relational_base, relational_candidate,
                           v5_base, v5_candidate):
    scene_ids = [str(value) for value in reference["rel_scene_ids"].tolist()]
    target_names = [str(value) for value in reference["rel_target_names"].tolist()]
    strata = [str(value) for value in reference["rel_strata"].tolist()]
    prompt_ids = [str(value) for value in reference["rel_prompt_ids"].tolist()]
    v5_targets = [str(value) for value in reference["v5_targets"].tolist()]
    return [
        aggregate_metrics(
            reference["rel_gt"], relational_base[:, generation],
            relational_candidate[:, generation], reference["rel_instance_ids"],
            reference["rel_category_ids"], reference["rel_target_instance_ids"],
            scene_ids, target_names, strata, prompt_ids, reference["v5_gt"],
            v5_base[:, generation], v5_candidate[:, generation], v5_targets,
        )
        for generation in range(GENERATIONS)
    ]


def pooled_metrics(generations):
    relational_cases = []
    prompt_pairs = []
    v5_cases = []
    for generation, metrics in enumerate(generations):
        relational_cases.extend({**row, "generation": generation}
                                for row in metrics["relational_cases"])
        prompt_pairs.extend({**row, "generation": generation}
                            for row in metrics["prompt_pairs"])
        v5_cases.extend({**row, "generation": generation}
                        for row in metrics["v5_replay_cases"])

    def target_summary(rows):
        base = sum(float(row["target_mae"]["base"]) for row in rows) / len(rows)
        candidate = sum(float(row["target_mae"]["candidate"]) for row in rows) / len(rows)
        return {
            "target_mae_base": base,
            "target_mae_candidate": candidate,
            "target_mae_relative_change": (candidate - base) / max(base, 1e-12),
            "case_win_rate": sum(
                row["target_mae"]["candidate"] < row["target_mae"]["base"]
                for row in rows
            ) / len(rows),
        }

    high_desk_cases = [row for row in relational_cases
                       if row["training_stratum"] == "high_desk_chair_06"]
    if len(relational_cases) != 48 or len(high_desk_cases) != 12:
        raise ValueError("Teacher-v7 pooled relational inventory changed")
    if len(prompt_pairs) != 24 or len(v5_cases) != 9:
        raise ValueError("Teacher-v7 pooled pair/replay inventory changed")
    overall = target_summary(relational_cases)
    overall.update({
        "semantic_violation_base": sum(
            float(row["semantic"]["base"]["semantic_violation"])
            for row in relational_cases
        ) / len(relational_cases),
        "semantic_violation_candidate": sum(
            float(row["semantic"]["candidate"]["semantic_violation"])
            for row in relational_cases
        ) / len(relational_cases),
        "prompt_invariance_mse_base": sum(float(row["base_mse"]) for row in prompt_pairs)
        / len(prompt_pairs),
        "prompt_invariance_mse_candidate": sum(float(row["candidate_mse"]) for row in prompt_pairs)
        / len(prompt_pairs),
        "v5_replay_mae_base": sum(float(row["base_mae"]) for row in v5_cases)
        / len(v5_cases),
        "v5_replay_mae_candidate": sum(float(row["candidate_mae"]) for row in v5_cases)
        / len(v5_cases),
    })
    overall["v5_replay_relative_degradation"] = (
        overall["v5_replay_mae_candidate"] - overall["v5_replay_mae_base"]
    ) / max(overall["v5_replay_mae_base"], 1e-12)
    return {
        "relational_cases": relational_cases,
        "prompt_pairs": prompt_pairs,
        "v5_replay_cases": v5_cases,
        "high_desk": target_summary(high_desk_cases),
        "overall": overall,
    }
