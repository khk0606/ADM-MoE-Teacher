#!/usr/bin/env python3
"""CPU-only unit/source contract for the few-shot CDM stage."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Local packaging host may not have PyTorch.
    torch = None

from fewshot_cdm_common import (
    CONTACT_JOINTS,
    build_sparse_contact_weights,
    checkpoint_selection_gate,
    compute_distance_map,
    denormalize_contact,
    distance_to_affordance,
    instance_scores,
    load_split,
    normalize_contact,
    semantic_checkpoint_gate,
    target_balanced_batches,
)
from fewshot_cdm_rollout_cache import (
    CONTRACT_SCHEMA,
    atomic_save_prediction,
    atomic_write_json,
    candidate_result_path,
    ensure_contract,
    fingerprint_rows,
    load_bound_candidate_result,
    load_prediction,
    prediction_path,
    sha256_array,
)
from fewshot_cdm_v5_semantics import (
    FORBIDDEN_PROMPT_WORDS,
    PROMPT_BY_TARGET,
    PROMPT_POLICY_ID,
    prompt_for_target,
)

if torch is not None:
    from evaluate_fewshot_cdm import (
        sample_contact_deterministic,
        stable_rollout_seeds,
    )
    from fewshot_cdm_lora import (
        LoRALinear,
        install_lora,
        lora_named_parameters,
        lora_parameter_energy,
        merged_legacy_state_dict,
        set_frozen_base_eval_lora_train,
        set_lora_enabled,
    )
    from train_fewshot_cdm import (
        masked_point_xstart_loss,
        semantic_contact_loss,
        weighted_xstart_loss,
    )
    from fewshot_cdm_v5_semantics import sit_multicandidate_loss


HERE = Path(__file__).resolve().parent


def test_distance_and_normalization() -> None:
    scene = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    motion = np.zeros((2, 22, 3), dtype=np.float32)
    for joint in CONTACT_JOINTS:
        motion[:, joint, 0] = np.asarray([0.0, 2.0], dtype=np.float32)
    distance = compute_distance_map(scene, motion, chunk_size=1)
    assert distance.shape == (2, 6)
    assert np.allclose(distance, np.asarray([[0.0] * 6, [1.0] * 6]))
    affordance = distance_to_affordance(distance, 0.8)
    mean = np.asarray([[0.2] * 6], dtype=np.float32)
    std = np.asarray([[0.3] * 6], dtype=np.float32)
    normalized = normalize_contact(affordance, mean, std)
    recovered = denormalize_contact(normalized, mean, std)
    assert np.allclose(recovered, affordance, atol=1e-6)
    print("[PASS] full-motion six-joint GT and mean/std round trip")


def test_split_and_replay_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="fewshot_cdm_contract_") as temp:
        path = Path(temp) / "split.json"
        train = ["room_0001_sit_chair_0002", "room_0003_lie_bed_0002", "room_0004_interact_whiteboard_0002"]
        test = ["room_0001_sit_chair_0003", "room_0003_lie_bed_0005", "room_0004_interact_whiteboard_0003"]
        targets = {
            train[0]: "chair", train[1]: "bed", train[2]: "whiteboard",
            test[0]: "chair", test[1]: "bed", test[2]: "whiteboard",
        }
        payload = {
            "schema": "affordance_source_disjoint_split_v1",
            "checks": {
                "source_components_disjoint": True,
                "original_motion_ids_disjoint": True,
                "multistart_excluded_from_cdm": True,
                "object_names_absent_from_new_prompts": True,
            },
            "cdm_fewshot": {"train": train, "test": test},
            "samples": [
                {
                    "sample_id": sample_id,
                    "target": targets[sample_id],
                    "stage": "base",
                    "split": "train" if sample_id in train else "test",
                }
                for sample_id in train + test
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        split = load_split(path)
        rows = [row for row in split["samples"] if row["split"] == "train"]
        batches = target_balanced_batches(rows, steps=9, seed=7)
        assert len(batches) == 9
        assert all(len(batch) == 3 for batch in batches)
        assert all(set(batch) == set(train) for batch in batches)
        print("[PASS] disjoint Base-only split and one-per-target Chair replay")

        v5_batches = target_balanced_batches(
            rows,
            steps=4,
            seed=7,
            replay_counts={"chair": 3, "bed": 1, "whiteboard": 1},
        )
        assert all(len(batch) == 5 for batch in v5_batches)
        for batch in v5_batches:
            names = [targets[sample_id] for sample_id in batch]
            assert names.count("chair") == 3
            assert names.count("bed") == 1
            assert names.count("whiteboard") == 1
        print("[PASS] v5 Chair3+Bed1+Whiteboard1 replay contract")


def test_v5_prompt_and_sit_prior_contract() -> None:
    assert PROMPT_POLICY_ID == "object_agnostic_action_v5"
    assert set(PROMPT_BY_TARGET) == {"chair", "bed", "whiteboard"}
    for target, prompt in PROMPT_BY_TARGET.items():
        assert prompt_for_target(target) == prompt
        assert not any(word in prompt.lower() for word in FORBIDDEN_PROMPT_WORDS)
    assert PROMPT_BY_TARGET["chair"] == "Sit somewhere."
    assert PROMPT_BY_TARGET["bed"] == "Lie down somewhere."
    assert "right hand" in PROMPT_BY_TARGET["whiteboard"].lower()
    if torch is None:
        print("[SKIP] v5 differentiable Sit prior (PyTorch unavailable)")
        return
    ids = torch.tensor([[1] * 6 + [2] * 6])
    mean = torch.zeros((1, 1, 6))
    std = torch.ones((1, 1, 6))
    for initial_value in (-4.0, 4.0):
        prediction = torch.full(
            (1, 12, 6), initial_value, requires_grad=True
        )
        result = sit_multicandidate_loss(
            prediction,
            ids,
            ["chair"],
            mean,
            std,
            bed_any_min=0.10,
            bed_any_max=0.35,
            bed_pelvis_min=0.05,
            bed_pelvis_max=0.25,
            chair_bed_margin=0.10,
        )
        loss = result["total"]
        loss.backward()
        assert torch.isfinite(loss)
        assert prediction.grad is not None
        assert float(prediction.grad.abs().sum()) > 0.0
        for metric in (
            "chair_any_top10",
            "bed_any_top10",
            "chair_pelvis_top10",
            "bed_pelvis_top10",
        ):
            assert 0.0 <= float(result[metric]) <= 1.0
    print("[PASS] exact v5 prompts and weak differentiable Sit multi-candidate prior")


def test_checkpoint_gate() -> None:
    initial = {
        "per_target_mean": {"chair": 1.0, "bed": 2.0, "whiteboard": 3.0}
    }
    passing = {
        "per_target_mean": {"chair": 1.04, "bed": 1.8, "whiteboard": 2.7}
    }
    decision = checkpoint_selection_gate(initial, passing, 0.05, 0.05)
    assert decision["passed"] is True
    assert decision["checks"] == {
        "bed_train_grid_improved": True,
        "whiteboard_train_grid_improved": True,
        "chair_train_replay_retained": True,
    }
    chair_failure = {
        "per_target_mean": {"chair": 1.11, "bed": 1.8, "whiteboard": 2.7}
    }
    assert checkpoint_selection_gate(initial, chair_failure, 0.05, 0.05)[
        "passed"
    ] is False
    print("[PASS] target-specific train-only checkpoint gate")


def test_zero_init_lora_contract() -> None:
    if torch is None:
        print("[SKIP] zero-init LoRA tensor contract (PyTorch unavailable)")
        return

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(4, 5),
                torch.nn.GELU(),
                torch.nn.Linear(5, 3),
            )

        def forward(self, value):
            return self.net(value)

    torch.manual_seed(3)
    model = Tiny()
    value = torch.randn(7, 4)
    original = model(value).detach().clone()
    original_state = {
        key: tensor.detach().clone() for key, tensor in model.state_dict().items()
    }
    names = install_lora(model, rank=2, alpha=4.0)
    assert names == ["net.0", "net.2"]
    assert torch.equal(model(value), original)
    model.train()
    set_frozen_base_eval_lora_train(model)
    assert model.training is False
    assert all(
        module.dropout.training
        for module in model.modules()
        if isinstance(module, LoRALinear)
    )
    trainable = lora_named_parameters(model)
    assert trainable
    assert all(parameter.requires_grad for parameter in trainable.values())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in trainable
    )
    loss = model(value).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for name, parameter in trainable.items()
        if name.endswith("lora_B")
    )
    with torch.no_grad():
        for name, parameter in trainable.items():
            if name.endswith("lora_B"):
                parameter.add_(0.01)
    assert float(lora_parameter_energy(model)) > 0.0
    adapted = model(value).detach()
    assert not torch.equal(adapted, original)
    merged = merged_legacy_state_dict(model)
    legacy = Tiny()
    legacy.load_state_dict(merged, strict=True)
    assert torch.allclose(legacy(value), adapted, atol=1e-6, rtol=1e-6)
    set_lora_enabled(model, False)
    assert torch.equal(model(value), original)
    assert not torch.equal(merged["net.0.weight"], original_state["net.0.weight"])
    assert not torch.equal(merged["net.2.weight"], original_state["net.2.weight"])
    assert torch.equal(merged["net.0.bias"], original_state["net.0.bias"])
    assert torch.equal(merged["net.2.bias"], original_state["net.2.bias"])
    print("[PASS] zero-init frozen-base LoRA, gradient, and merged export contract")


def test_semantic_loss_prevents_sparse_zero_collapse() -> None:
    if torch is None:
        print("[SKIP] semantic sparse-contact tensor test (PyTorch unavailable)")
        return
    batch, points, channels = 2, 12, 6
    instance_ids = torch.tensor(
        [[1] * 4 + [2] * 4 + [3] * 4] * batch, dtype=torch.long
    )
    target_ids = torch.tensor([2, 3], dtype=torch.long)
    targets = ["bed", "whiteboard"]
    gt = torch.zeros(batch, points, channels)
    gt[0, 4:8, 0] = 0.95
    gt[0, 4:8, 5] = 0.90
    gt[1, 8:12, 2] = 0.95
    prediction = torch.zeros_like(gt, requires_grad=True)
    mean = torch.zeros(1, 1, channels)
    std = torch.ones(1, 1, channels)

    def compute(value):
        return semantic_contact_loss(
            value,
            gt,
            instance_ids,
            target_ids,
            targets,
            mean,
            std,
            0.7,
            0.1,
            1.0,
            1.0,
            0.25,
            2.0,
            0.1,
        )["total"]

    collapsed = compute(prediction)
    collapsed.backward()
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0
    aligned = torch.full_like(gt, -1.0)
    aligned[0, 4:8, 0] = 1.0
    aligned[0, 4:8, 5] = 1.0
    aligned[1, 8:12, 2] = 1.0
    assert float(compute(aligned)) < float(collapsed.detach())
    print("[PASS] foreground/Dice/ranking loss rejects sparse zero-contact collapse")


def test_sparse_target_weight_contract() -> None:
    gt = np.zeros((8192, 6), dtype=np.float32)
    instance_ids = np.zeros(8192, dtype=np.int64)
    instance_ids[100:300] = 1
    instance_ids[300:500] = 2
    instance_ids[500:700] = 3
    gt[350:375, 0] = 0.9
    gt[550:575, 5] = 0.95

    chair = build_sparse_contact_weights(gt, instance_ids, 1, "chair", 4.0, 16.0)
    assert np.array_equal(chair, np.ones_like(chair))
    bed = build_sparse_contact_weights(gt, instance_ids, 2, "bed", 4.0, 16.0)
    assert float(bed[0, 0]) == 1.0
    assert float(bed[300, 0]) == 5.0
    assert float(bed[350, 0]) == 21.0
    assert float(bed[350, 1]) == 5.0
    assert np.all(bed[instance_ids == 3] == 1.0)
    whiteboard = build_sparse_contact_weights(
        gt, instance_ids, 3, "whiteboard", 4.0, 16.0
    )
    assert float(whiteboard[550, 5]) == 21.0
    assert np.all(whiteboard[instance_ids == 2] == 1.0)

    if torch is not None:
        prediction = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        target = torch.zeros_like(prediction)
        uniform = torch.ones_like(prediction)
        legacy = (prediction - target).square().mean(dim=(1, 2))
        weighted = weighted_xstart_loss(prediction, target, uniform)
        assert torch.equal(legacy, weighted)
        emphasized = uniform.clone()
        emphasized[0, 0, 0] = 9.0
        assert float(
            weighted_xstart_loss(prediction, target, emphasized).item()
        ) != float(legacy.item())
        print("[PASS] direct START_X weighted MSE and uniform parity")
    else:
        print("[SKIP] direct START_X tensor parity (PyTorch unavailable)")
    print("[PASS] sparse novel-target weights preserve Chair legacy supervision")


def test_chair_region_multinoise_loss_contract() -> None:
    if torch is None:
        print("[SKIP] Chair-region multi-noise tensor contract (PyTorch unavailable)")
        return
    prediction = torch.tensor(
        [[[1.0, 2.0], [30.0, 40.0], [5.0, 6.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    chair_mask = torch.tensor([[True, False, True]])
    loss = masked_point_xstart_loss(prediction, target, chair_mask)
    expected = torch.tensor([(1.0 + 4.0 + 25.0 + 36.0) / 4.0])
    assert torch.allclose(loss.detach(), expected)
    loss.mean().backward()
    assert prediction.grad is not None
    assert torch.equal(prediction.grad[:, 1], torch.zeros_like(prediction.grad[:, 1]))
    assert float(prediction.grad[:, [0, 2]].abs().sum()) > 0.0
    print("[PASS] Chair-region loss excludes the weak Bed candidate region")


def test_semantic_checkpoint_gate() -> None:
    semantic = {
        "per_target": {
            "chair": {
                "f1_at_0_7": 0.8,
                "any_joint_dominance_rate": 0.95,
                "pelvis_dominance_rate": 0.95,
                "right_wrist_dominance_rate": 0.0,
                "sit_chair_primary_rate": 0.95,
                "sit_bed_candidate_visible_rate": 0.95,
            },
            "bed": {
                "f1_at_0_7": 0.45,
                "any_joint_dominance_rate": 1.0,
                "pelvis_dominance_rate": 1.0,
                "right_wrist_dominance_rate": 0.0,
                "sit_chair_primary_rate": 1.0,
                "sit_bed_candidate_visible_rate": 1.0,
            },
            "whiteboard": {
                "f1_at_0_7": 0.55,
                "any_joint_dominance_rate": 1.0,
                "pelvis_dominance_rate": 0.0,
                "right_wrist_dominance_rate": 1.0,
                "sit_chair_primary_rate": 1.0,
                "sit_bed_candidate_visible_rate": 1.0,
            },
        }
    }
    decision = semantic_checkpoint_gate(semantic, 0.30, 1.0, 0.90, 0.80)
    assert decision["passed"] is True
    assert all(decision["checks"].values())
    semantic["per_target"]["chair"]["sit_bed_candidate_visible_rate"] = 0.85
    assert semantic_checkpoint_gate(semantic, 0.30, 1.0, 0.90, 0.80)[
        "passed"
    ] is True
    semantic["per_target"]["chair"]["sit_bed_candidate_visible_rate"] = 0.75
    weak_failure = semantic_checkpoint_gate(semantic, 0.30, 1.0, 0.90, 0.80)
    assert weak_failure["passed"] is False
    assert weak_failure["checks"][
        "sit_train_bed_candidate_is_visible_and_bounded"
    ] is False
    semantic["per_target"]["chair"]["sit_bed_candidate_visible_rate"] = 0.95
    semantic["per_target"]["whiteboard"]["f1_at_0_7"] = 0.1
    assert semantic_checkpoint_gate(semantic, 0.30, 1.0, 0.90, 0.80)[
        "passed"
    ] is False
    print("[PASS] train-only activity and target-dominance semantic gate")


def test_instance_scoring() -> None:
    affordance = np.zeros((8192, 6), dtype=np.float32)
    instance_ids = np.zeros(8192, dtype=np.int64)
    instance_ids[100:200] = 1
    instance_ids[200:300] = 2
    instance_ids[300:400] = 3
    affordance[200:300] = 0.9
    scores = instance_scores(affordance, instance_ids, "any_joint")
    assert max(("chair", "bed", "whiteboard"), key=lambda v: scores[v]) == "bed"
    affordance[:] = 0.0
    affordance[300:400, 5] = 0.95
    wrist_scores = instance_scores(affordance, instance_ids, "right_wrist")
    assert max(("chair", "bed", "whiteboard"), key=lambda v: wrist_scores[v]) == "whiteboard"
    print("[PASS] target-instance top-10% Base ADM scoring")


def test_full_rollout_pairing_contract() -> None:
    if torch is None:
        print("[SKIP] full-rollout RNG pairing test (PyTorch unavailable)")
        return

    class DummyDiffusion:
        def p_sample_loop(self, model, shape, noise, **kwargs):
            del model, shape, kwargs
            return noise + torch.randn_like(noise) + torch.randn_like(noise)

    row = {
        "xyz": np.zeros((8192, 3), dtype=np.float32),
        "feat": np.zeros((8192, 3), dtype=np.float32),
        "text": "object-agnostic action",
    }
    seeds = stable_rollout_seeds(17, "train", "sample_b", 2)
    assert seeds == stable_rollout_seeds(17, "train", "sample_b", 2)
    assert seeds != stable_rollout_seeds(17, "train", "sample_a", 2)
    first = sample_contact_deterministic(
        object(), DummyDiffusion(), row, seeds[0], seeds[1], "cpu"
    )
    torch.manual_seed(999)
    _ = torch.randn(37)
    second = sample_contact_deterministic(
        object(), DummyDiffusion(), row, seeds[0], seeds[1], "cpu"
    )
    assert np.array_equal(first, second)

    torch.manual_seed(1234)
    expected_before = torch.rand(1)
    expected_after = torch.rand(1)
    torch.manual_seed(1234)
    actual_before = torch.rand(1)
    _ = sample_contact_deterministic(
        object(), DummyDiffusion(), row, seeds[0], seeds[1], "cpu"
    )
    actual_after = torch.rand(1)
    assert torch.equal(expected_before, actual_before)
    assert torch.equal(expected_after, actual_after)
    print("[PASS] full reverse-noise pairing and caller RNG preservation")


def test_crash_safe_rollout_cache_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="fewshot_rollout_cache_") as temp:
        root = Path(temp)
        rows = {
            "sample_a": {
                "scene_id": "room_0001",
                "target": "chair",
                "target_instance_id": 1,
                "text": "object-agnostic action",
                "gt": np.zeros((8, 6), dtype=np.float32),
                "xyz": np.zeros((8, 3), dtype=np.float32),
                "feat": np.ones((8, 3), dtype=np.float32),
                "instance_ids": np.arange(8, dtype=np.int64),
                "source_indices": np.arange(8, dtype=np.int64),
            }
        }
        fingerprints = fingerprint_rows(rows, ["sample_a"])
        assert fingerprints["sample_a"]["gt_sha256"] == sha256_array(
            rows["sample_a"]["gt"]
        )
        contract = {
            "schema": CONTRACT_SCHEMA,
            "split_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "sample_fingerprints": fingerprints,
            "k_samples": 5,
        }
        first_hash = ensure_contract(root, contract)
        assert ensure_contract(root, contract) == first_hash
        incompatible = dict(contract)
        incompatible["k_samples"] = 6
        try:
            ensure_contract(root, incompatible)
        except RuntimeError as error:
            assert "incompatible rollout cache" in str(error)
        else:
            raise AssertionError("incompatible rollout cache was silently reused")

        prediction = np.arange(8192 * 6, dtype=np.float32).reshape(8192, 6)
        prediction_file = prediction_path(root, "original", "sample_a", 0)
        atomic_save_prediction(prediction_file, prediction)
        assert np.array_equal(load_prediction(prediction_file), prediction)
        candidate_file = prediction_path(
            root, "candidate", "sample_a", 4, candidate_step=3500
        )
        atomic_save_prediction(candidate_file, prediction + 1.0)
        assert np.array_equal(load_prediction(candidate_file), prediction + 1.0)

        result_file = candidate_result_path(root, 3500)
        result = {
            "contract_sha256": first_hash,
            "checkpoint_sha256": "c" * 64,
            "step": 3500,
            "passed": False,
        }
        atomic_write_json(result_file, result)
        assert load_bound_candidate_result(
            result_file, first_hash, "c" * 64, 3500
        ) == result
        try:
            load_bound_candidate_result(result_file, first_hash, "d" * 64, 3500)
        except RuntimeError as error:
            assert "checkpoint mismatch" in str(error)
        else:
            raise AssertionError("candidate result checkpoint mismatch was ignored")
    print("[PASS] crash-safe per-draw rollout cache and strict resume contract")


def test_source_guards() -> None:
    train_source = (HERE / "train_fewshot_cdm.py").read_text(encoding="utf-8")
    eval_source = (HERE / "evaluate_fewshot_cdm.py").read_text(encoding="utf-8")
    lora_source = (HERE / "fewshot_cdm_lora.py").read_text(encoding="utf-8")
    gt_source = (HERE / "generate_fewshot_cdm_gt.py").read_text(encoding="utf-8")
    audit_source = (HERE / "audit_fewshot_cdm_train_rollout.py").read_text(
        encoding="utf-8"
    )
    assert "one-sample checkpoint" in train_source
    assert "target_balanced_batches" in train_source
    assert "chair3_bed1_whiteboard1_shuffled_cycles" in train_source
    assert "chair_region_multinoise_teacher_v5r4" in train_source
    assert "rollout_aligned_sparse_semantic_multinoise_start_x_v5r4" in train_source
    assert "one_step_shortlist_then_train_full_rollout" in train_source
    assert "select_candidate_by_train_rollout" in train_source
    assert '"partition": "train_complete"' in train_source
    assert '"seed_partition": "train_audit"' in train_source
    assert "strict v5r4 requires --candidate-rollout-k >= 5" in train_source
    assert "--stop-after-one-step" in train_source
    assert "fewshot_cdm_v5_semantics" in train_source
    assert "--sit-bed-candidate-min-rate" in train_source
    assert "per_draw_sit_chair_primary" in train_source
    assert "per_draw_sit_bed_candidate_visible" in train_source
    assert "sit_per_draw_chair_primary" in train_source
    assert "sit_per_draw_bed_candidate_visible_and_bounded" in train_source
    assert '"weight": args.sit_multicandidate_weight' in train_source
    assert '"sit_bed_candidate_min_rate": args.sit_bed_candidate_min_rate' in train_source
    assert "weak Sit-Bed coverage must be lower than strict Chair dominance" in train_source
    assert "DIAGNOSTIC-CHECKPOINT" in train_source
    assert "diagnostic_checkpoints.json" in train_source
    assert "train_only_diagnostic_not_promoted" in train_source
    assert "--sit-bed-candidate-min-rate" in audit_source
    assert "sit_ensemble_bed_candidate_rate" in audit_source
    assert ">= args.sit_bed_candidate_min_rate" in audit_source
    assert '"sit_contract"' in audit_source
    assert "right_wrist_native_index_5" in train_source
    assert "sample_contact_deterministic" in train_source
    assert "No shortlisted checkpoint passed train-only full-rollout" in train_source
    assert "--rollout-only" in train_source
    assert "rollout_cache" in train_source
    assert "ROLLOUT-PROGRESS" in train_source
    assert "atomic_save_prediction" in train_source
    assert "fingerprint_rows" in train_source
    assert "heldout_sample_tensors_read" in train_source
    assert "semantic_contact_loss" in train_source
    assert "high_timestep_extra_forward" in train_source
    assert "predict_xstart" in train_source
    assert "weighted_xstart_loss" in train_source
    assert "masked_point_xstart_loss" in train_source
    assert "chair_high_timestep" in train_source
    assert "chair_high_teacher_loss" in train_source
    assert "bed_candidate_region_excluded_from_chair_teacher" in train_source
    assert "assert_chair_uniform_loss_parity" in train_source
    assert "test_partition_read_during_training" in train_source
    assert '"train",\n        mean,\n        std,' in train_source
    assert 'split["cdm_fewshot"]["test"]' in eval_source
    assert "paired" in eval_source.lower()
    assert "may_proceed_to_moe_iiw" in eval_source
    assert "training_schema_is_strict_v5r4" in eval_source
    assert "validate_v5_training_provenance" in eval_source
    assert "train_only_full_rollout_selected_checkpoint" in eval_source
    assert "--train-rollout-audit" in eval_source
    assert "validate_train_rollout_audit" in eval_source
    assert eval_source.index("validate_train_rollout_audit(", 5000) < eval_source.index(
        "rows = load_test_rows"
    )
    assert "sample_contact_deterministic" in eval_source
    assert "initial_xT_and_all_reverse_step_noise_paired" in eval_source
    assert "sit_per_draw_chair_primary_rate_pass" in eval_source
    assert "sit_per_draw_bed_candidate_rate_pass" in eval_source
    assert "history_affordance_v1_fewshot_cdm_train_rollout_audit_v6" in audit_source
    assert '"diffusion_steps": 500' in audit_source
    assert '"novel_min_f1": 0.30' in audit_source
    assert '"dominance_margin": 0.01' in audit_source
    assert "strict train rollout audit gate values changed" in audit_source
    assert "paired_rollout_affordance_npz_v2" in audit_source
    assert "initial_noise_seeds=saved_seed_pairs[:, :, 0]" in audit_source
    assert "reverse_noise_seeds=saved_seed_pairs[:, :, 1]" in audit_source
    assert "per_draw_target_region_metrics" in audit_source
    assert "full_trajectory_repeatability_canary" in audit_source
    assert "training_schema_is_strict_v5r4" in audit_source
    assert "train_only_full_rollout_selected_checkpoint" in audit_source
    assert "strict development evaluation gate values changed" in eval_source
    assert "paired_rollout_affordance_npz_v2" in eval_source
    assert "initial_noise_seeds=saved_seed_pairs[:, :, 0]" in eval_source
    assert "reverse_noise_seeds=saved_seed_pairs[:, :, 1]" in eval_source
    assert 'raise SystemExit(2)' in audit_source
    assert "merged_legacy_partial_state_dict" in lora_source
    assert "type(child) is torch.nn.Linear" in lora_source
    assert "global_relative_l2sp_to_original_cdm" not in train_source
    assert "gt_file_from_entry" in gt_source
    assert "cmdm_motion_input" not in train_source
    print("[PASS] original-init, train/test isolation, and MoE-IIW gate source contract")


def main() -> None:
    test_distance_and_normalization()
    test_split_and_replay_contract()
    test_v5_prompt_and_sit_prior_contract()
    test_checkpoint_gate()
    test_zero_init_lora_contract()
    test_semantic_loss_prevents_sparse_zero_collapse()
    test_sparse_target_weight_contract()
    test_chair_region_multinoise_loss_contract()
    test_semantic_checkpoint_gate()
    test_instance_scoring()
    test_full_rollout_pairing_contract()
    test_crash_safe_rollout_cache_contract()
    test_source_guards()
    print("[PASS] few-shot CDM pipeline unit contract")


if __name__ == "__main__":
    main()
