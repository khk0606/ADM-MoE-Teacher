#!/usr/bin/env python3
"""Print the sealed v9.2 response evidence needed for the next objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def means(panel):
    result = {}
    for name in ("bed_01", "chair_01", "chair_06"):
        rows = [item["instances"][name] for item in panel["per_prompt_metrics"]]
        result[name] = {
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in ("topk_overlap", "soft_recall", "active_support_mae")
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.report.expanduser().resolve().read_text(encoding="utf-8"))
    if value.get("schema") != "relational_teacher_v92_hotspot_trust_response_v1":
        raise ValueError("not a Teacher-v9.2 hotspot/trust report")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("Teacher-v9.2 candidate inventory changed")
    print("=== V9.2 EXACT RESPONSE DIAGNOSIS ===")
    hashes = []
    for row in candidates:
        before = row["before"]
        after = row["after"]
        hashes.append(str(row["prediction_sha256"]))
        print("\nCANDIDATE", row["name"], "eligible", row["eligible"])
        print("failed", row["failed_checks"])
        print("prediction_sha256", row["prediction_sha256"])
        print(
            "loss active/margin/listwise",
            (before["active_support_macro"], after["active_support_macro"]),
            (before["hotspot_margin_macro"], after["hotspot_margin_macro"]),
            (before["hotspot_listwise_macro"], after["hotspot_listwise_macro"]),
        )
        print(
            "trust prompt negative(mean/max)",
            after["background_trust"],
            (before["prompt_invariance"], after["prompt_invariance"]),
            (before["negative_mean"], after["negative_mean"]),
            (before["negative_max"], after["negative_max"]),
        )
        print(
            "v5 mean",
            sum(row["base_v5_dense"]) / 3.0,
            "->",
            sum(row["candidate_v5_dense"]) / 3.0,
        )
        before_instances = means(before)
        after_instances = means(after)
        for name in ("bed_01", "chair_01", "chair_06"):
            print(name, before_instances[name], "->", after_instances[name])
        print("gradient first/last", row["update_log"][0]["gradient_l2_before_clip"], row["update_log"][-1]["gradient_l2_before_clip"])
    print("\nunique_prediction_hashes", len(set(hashes)), "/", len(hashes))
    print("status", value["status"], "selected", value["selected_candidate"])


if __name__ == "__main__":
    main()

