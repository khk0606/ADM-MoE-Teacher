#!/usr/bin/env python3
"""CPU-only Gate 0A verifier for one explicit v5r4 experiment directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_DIR = Path(__file__).resolve().parent
for value in (REPO_ROOT, PREPARE_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from base_teacher_contract import (  # noqa: E402
    atomic_write_json,
    validate_v5r4_quality_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--stats-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.report.expanduser().resolve().exists():
        raise FileExistsError(
            "refusing to overwrite an existing Gate 0A evidence seal: "
            + str(args.report.expanduser().resolve())
        )
    root = args.experiment_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected_name = "fewshot_cdm_chair23_bed2_whiteboard12_v5r4"
    if root.name != expected_name:
        raise ValueError(
            "experiment root must be the explicit v5r4 directory named "
            + expected_name
        )
    result = validate_v5r4_quality_chain(
        fewshot_checkpoint=root / "fewshot_cdm.pt",
        original_checkpoint=args.original_checkpoint,
        train_summary_file=root / "summary.json",
        shortlist_status_file=root / "shortlist_status.json",
        rollout_selection_file=root / "rollout_candidate_selection.json",
        train_rollout_audit_file=root / "train_full_rollout_k5" / "summary.json",
        train_rollout_predictions_file=(
            root
            / "train_full_rollout_k5"
            / "train_rollout_predictions.npz"
        ),
        development_summary_file=(
            root / "heldout_base_adm_eval" / "summary.json"
        ),
        development_predictions_file=(
            root / "heldout_base_adm_eval" / "predictions.npz"
        ),
        split_file=args.split,
        stats_file=args.stats_file,
        minimum_k_draws=5,
        expected_diffusion_steps=500,
        runtime_repo_root=REPO_ROOT,
        dataset_root=args.dataset_root,
    )
    report = {
        "schema": "history_affordance_v2_gate0a_evidence_report_v1",
        "status": "PASS",
        "may_export_base_teacher": True,
        "experiment_root": str(root),
        "quality_chain": result,
    }
    atomic_write_json(args.report, report)
    print("[PASS] Gate 0A v5r4 evidence chain")
    print("[OK] few-shot checkpoint SHA-256: " + result["sha256"]["fewshot_checkpoint"])
    print("[OK] development draws: " + str(result["development_k_draws"]))
    print("[OK] evidence-set SHA-256: " + result["evidence_set_sha256"])
    print("[OK] report: " + str(args.report.expanduser().resolve()))


if __name__ == "__main__":
    main()
