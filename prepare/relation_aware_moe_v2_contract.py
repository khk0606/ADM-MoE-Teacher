#!/usr/bin/env python3
"""Pure contracts for the leakage-safe relation-aware map MoE v2.

This module intentionally imports neither Torch nor project models, allowing
the data/checkpoint policy to be tested before CUDA work starts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "relation_aware_map_moe_v2"
FORMAT_VERSION = 4
STATUS = "RESEARCH_ONLY"
POINT_COUNT = 8192
CONTACT_DIM = 6
ALPHA_UP = 0.15
ALPHA_DOWN = 0.10
WEIGHT_RANGE = (0.0, 1.0)
IDENTITY_WEIGHT = 0.5
RELATION_BRANCH_COEFFICIENT = 0.60
HISTORY_BRANCH_COEFFICIENT = 0.40
BRANCH_EXPERT_COUNT = 2
OBSERVED_HISTORY_FRAMES = 8
OBSERVED_HISTORY_DIM = 6
FUTURE_SUFFIX_START_FRAME = OBSERVED_HISTORY_FRAMES
OBSERVED_HISTORY_CHANNELS = (
    "current_x",
    "current_y",
    "velocity_x",
    "velocity_y",
    "heading_x",
    "heading_y",
)
OBSERVED_HISTORY_CONTRACT = {
    "shape": "[B,8,6]",
    "channels": list(OBSERVED_HISTORY_CHANNELS),
    "source": "strictly_observed_prefix_frames_0_through_7",
    "ordering": "oldest_to_newest",
    "velocity": "backward_observed_only_finite_difference",
    "heading": "same_frame_body_heading_unit_xy",
    "future_suffix_start_frame": FUTURE_SUFFIX_START_FRAME,
    "future_frames_allowed": False,
    "gt_motion_allowed": False,
    "required_for_supported_prompts": True,
}
TWO_BRANCH_MOE_METADATA = {
    "topology": "independent_relation_and_history_two_expert_moe_v1",
    "relation_experts": BRANCH_EXPERT_COUNT,
    "history_experts": BRANCH_EXPERT_COUNT,
    "relation_coefficient": RELATION_BRANCH_COEFFICIENT,
    "history_coefficient": HISTORY_BRANCH_COEFFICIENT,
    "coefficients_trainable": False,
    "fusion": "pointwise_weighted_signed_correction_before_bounded_residual",
    "direct_teacher_product": False,
}
SOURCE_IDENTITY_KEYS = (
    "source_collection_id",
    "source_group_id",
    "source_sample_id",
)
SOURCE_MOTION_HASH_KEY = "source_motion_sha256"

# These four values are the semantic order of the two learned relation routes.
# A checkpoint tensor is not meaningful unless its rows are bound to these
# exact prompt IDs, texts and CLIP tokenizer/model settings.
CANONICAL_PROMPT_IDS = ("sit_watch_v1", "sit_write_v1")
CANONICAL_PROMPT_TEXTS = (
    "Sit anywhere to watch.",
    "Sit anywhere to write.",
)
CANONICAL_CLIP_VERSION = "ViT-B/32"
CANONICAL_CLIP_MAX_LENGTH = 32
CANONICAL_PROMPT_BINDING_SCHEMA = SCHEMA + "_canonical_prompt_binding_v1"
SEALED_SUPPORT_FLOOR = 0.05
SEALED_SUPPORT_CEILING = 0.30

PROMPT_PURPOSE_CATEGORY = {
    "sit_watch_v1": "tv",
    "sit_write_v1": "desk",
}
PROMPT_EXPERT = {
    "sit_watch_v1": 0,
    "sit_write_v1": 1,
}
SUPPORTED_RELATION_PROMPT_IDS = tuple(PROMPT_PURPOSE_CATEGORY)
RELATION_SCOPE_POLICY = {
    "supported_prompt_ids": list(SUPPORTED_RELATION_PROMPT_IDS),
    "unsupported_prompt_policy": "exact_frozen_teacher_bypass_v1",
    "mixed_batch_supported": True,
}

# Only these relation bindings are currently physically/audit supported.  Bed
# rows are deliberately excluded rather than inventing a purpose label.
SEALED_PURPOSE_BINDINGS = {
    "room_0101": {
        "sit_watch_v1": {
            "purpose_instance_id": "tv_01",
            "target_instance_id": "chair_01",
            "source_group": "normal_chair",
        },
        "sit_write_v1": {
            "purpose_instance_id": "desk_01",
            "target_instance_id": "chair_06",
            "source_group": "high_desk_motion",
        },
    },
    "room_0102": {
        "sit_watch_v1": {
            "purpose_instance_id": "tv_01",
            "target_instance_id": "chair_05",
            "source_group": "normal_chair",
        },
        "sit_write_v1": {
            "purpose_instance_id": "desk_01",
            "target_instance_id": "chair_06",
            "source_group": "high_desk_motion",
        },
    },
}

FORWARD_INPUTS = (
    "scene_points",
    "CLIP_text_feature",
    "state",
    "observed_history",
    "teacher_affordance",
    "legacy_terminal_iiw",
)
FORBIDDEN_FORWARD_TOKENS = (
    "instance_id",
    "category_id",
    "target_id",
    "purpose_mask",
    "candidate_mask",
    "candidate_logits",
    "purpose_logits",
    "gt_anchor",
    "motion_id",
    "final_pelvis",
    "final_facing",
    "future_motion",
    "future_suffix",
    "target_map",
)

RELATION_FEATURE_NAMES = (
    "predicted_candidate_support",
    "predicted_sittable_probability",
    "predicted_purpose_probability",
    "predicted_purpose_confidence",
    "purpose_distance_normalized",
    "purpose_direction_x",
    "purpose_direction_y",
    "scene_centroid_distance_normalized",
    "scene_centroid_offset_x_normalized",
    "scene_centroid_offset_y_normalized",
    "purpose_vertical_offset_normalized",
    "purpose_distance_affinity",
    "soft_line_of_sight_clearance",
)
HISTORY_FEATURE_NAMES = (
    "observed_position_distance_normalized",
    "observed_bearing_forward",
    "observed_bearing_lateral",
    "observed_speed_normalized",
    "observed_velocity_toward_point_normalized",
    "observed_path_min_distance_normalized",
    "observed_approach_progress_normalized",
    "observed_net_motion_toward_point",
)

# A family ablation is only a diagnostic forward pass: the selected feature
# columns are deterministically permuted across points while every other input
# (and the canonical training forward) remains unchanged.  A non-zero map
# delta proves that the learned correction actually depends on that family.
RELATION_GEOMETRY_ABLATION_FAMILIES = (
    "distance",
    "direction",
    "visibility",
)
FAMILY_ABLATION_MAP_MAE_MIN = 1e-4
FAMILY_ABLATION_LOSS_TOLERANCE = 1e-6

MANDATORY_CHECKS = (
    "purpose_manifest_hash_bound",
    "source_reconstruction_exact",
    "forward_api_has_no_gt_or_ids",
    "observed_history_prefix_8_exact",
    "observed_history_future_suffix_disjoint",
    "observed_history_reconstruction_exact",
    "state_matches_history_endpoint",
    "future_suffix_target_recomputed_exact",
    "history_normalization_train_only",
    "locator_watch_tv_iou_pass",
    "locator_write_desk_iou_pass",
    "locator_prompt_swap_moves_anchor",
    "candidate_locator_iou_pass",
    "candidate_logits_never_supplied_externally",
    "teacher_v1_planner_clip_frozen",
    "legacy_v1_checkpoint_forward_replay_exact",
    "identity_initialization_bitwise_exact",
    "bounded_formula_exact",
    "weight_stays_in_unit_interval",
    "correct_relation_beats_identity",
    "correct_relation_beats_old_destructive_product",
    "correct_relation_beats_purpose_shuffle",
    "correct_relation_beats_state_shuffle",
    "relation_geometry_ablation_degrades",
    "watch_distance_dependency_pass",
    "watch_direction_dependency_pass",
    "watch_visibility_dependency_pass",
    "write_distance_dependency_pass",
    "write_direction_dependency_pass",
    "write_visibility_dependency_pass",
    "relation_history_branches_independent",
    "relation_router_two_experts",
    "history_router_two_experts",
    "fixed_fusion_0_6_0_4_exact",
    "fusion_coefficients_nontrainable",
    "relation_branch_counterfactual_dependency_pass",
    "history_branch_counterfactual_dependency_pass",
    "both_experts_used",
    "router_not_collapsed",
    "correction_not_saturated",
    "all_scene_prompt_subgroups_nonregressing",
    "all_scene_prompt_locator_quality_pass",
    "object_quality_pass",
    "negative_addition_bounded",
    "noncandidate_suppression_bounded",
    "v5_scope_dispatcher_exact",
    "room0201_unread_until_lock",
    "stable_pass_streak_met",
)
STABLE_PASS_CHECK = "stable_pass_streak_met"
SEALED_MAXIMUM_STEPS = 1500
SEALED_MINIMUM_STEPS = 500
SEALED_EVAL_EVERY = 100
SEALED_REQUIRED_CONSECUTIVE_PASSES = 3
SEALED_CONTRAST_EVERY = 4
SEALED_RELATION_GEOMETRY_CONTRAST_SCHEDULE = (
    "reverse_points_diagnostic",
    "ablate_distance",
    "ablate_direction",
    "ablate_visibility",
)
INSTANTANEOUS_CHECKS = tuple(
    name for name in MANDATORY_CHECKS if name != STABLE_PASS_CHECK
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_forward_input_contract(names: Sequence[str]) -> None:
    actual = tuple(str(value) for value in names)
    if actual != FORWARD_INPUTS:
        raise ValueError("v2 forward input order changed: " + repr(actual))
    lowered = "|".join(actual).lower()
    leaked = [token for token in FORBIDDEN_FORWARD_TOKENS if token in lowered]
    if leaked:
        raise ValueError("GT/identity token leaked into forward API: " + repr(leaked))


def validate_purpose_manifest(value: Mapping[str, object]) -> Dict[str, object]:
    """Validate exact stable-string bindings; numeric category IDs are banned."""
    if not isinstance(value, Mapping):
        raise TypeError("purpose manifest must be a mapping")
    if value.get("schema") != SCHEMA + "_purpose_manifest_v1":
        raise ValueError("purpose manifest schema changed")
    if value.get("format_version") != FORMAT_VERSION:
        raise ValueError("purpose manifest format version changed")
    if value.get("status") != "ORACLE_RELATION_RESEARCH_ONLY":
        raise ValueError("purpose manifest must remain research-only")
    if value.get("authorization") != "relation_aware_moe_v2_training_manifest_only":
        raise ValueError("purpose manifest authorization changed")
    if value.get("binding_count") != 4:
        raise ValueError("purpose manifest binding count changed")
    if value.get("source_scene_ids") != sorted(SEALED_PURPOSE_BINDINGS):
        raise ValueError("purpose manifest source scene set changed")
    if value.get("prompt_purpose_categories") != PROMPT_PURPOSE_CATEGORY:
        raise ValueError("purpose manifest prompt-purpose map changed")
    source_index_sha = str(value.get("source_index_sha256", "")).lower()
    if len(source_index_sha) != 64 or any(
        character not in "0123456789abcdef" for character in source_index_sha
    ):
        raise ValueError("purpose manifest source index SHA-256 is malformed")
    if value.get("numeric_category_ids_used_as_semantic_contract") is not False:
        raise ValueError("numeric category IDs cannot define semantics")
    if value.get("future_motion_used_for_state") is not False:
        raise ValueError("future motion cannot define the planner state")
    if value.get("room0201_arrays_read") is not False:
        raise ValueError("room_0201 arrays must remain unread before protocol lock")
    if value.get("failed_checks") != []:
        raise ValueError("purpose manifest contains failed checks")
    inference = value.get("inference_contract")
    if not isinstance(inference, Mapping):
        raise ValueError("purpose manifest inference contract is absent")
    assert_forward_input_contract(inference.get("forward_inputs", ()))
    if tuple(inference.get("derived_relation_features", ())) != RELATION_FEATURE_NAMES:
        raise ValueError("purpose manifest relation feature order changed")
    if (
        inference.get("purpose_source") != "predicted_from_scene_points_and_CLIP_text"
        or inference.get("ground_truth_or_identifiers_consumed_at_inference") != []
        or inference.get("sidecar_required_at_inference") is not False
    ):
        raise ValueError("purpose manifest permits inference-time oracle data")
    supervision = value.get("supervision_contract")
    required_supervision = {
        "bindings_are_training_only": True,
        "purpose_masks_are_training_only": True,
        "instance_and_category_ids_are_training_only": True,
        "numeric_object_ids_serialized": False,
        "ground_truth_masks_serialized": False,
        "canonical_semantics": "lowercase_category_strings",
    }
    if not isinstance(supervision, Mapping) or any(
        supervision.get(key) != expected
        for key, expected in required_supervision.items()
    ):
        raise ValueError("purpose manifest supervision contract changed")
    bindings = value.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise ValueError("purpose manifest must contain four scene/prompt bindings")
    observed = {}
    for raw in bindings:
        if not isinstance(raw, Mapping):
            raise ValueError("purpose binding is not an object")
        scene = str(raw.get("scene_id", ""))
        prompt = str(raw.get("prompt_id", ""))
        if (
            scene not in SEALED_PURPOSE_BINDINGS
            or prompt not in PROMPT_PURPOSE_CATEGORY
        ):
            raise ValueError("unexpected scene/prompt binding")
        if "purpose_numeric_id" in raw or "target_numeric_id" in raw:
            raise ValueError("numeric object IDs cannot be a semantic contract")
        if raw.get("supervision_only") is not True:
            raise ValueError("purpose binding must remain training-only")
        if raw.get("inference_feature") is not False:
            raise ValueError("purpose binding leaked into inference features")
        if not str(raw.get("prompt_text", "")).strip():
            raise ValueError("purpose binding prompt text is absent")
        expected = SEALED_PURPOSE_BINDINGS[scene][prompt]
        for key, expected_value in expected.items():
            if str(raw.get(key, "")) != expected_value:
                raise ValueError("{} {} {} changed".format(scene, prompt, key))
        if str(raw.get("purpose_category", "")) != PROMPT_PURPOSE_CATEGORY[prompt]:
            raise ValueError("prompt-purpose category changed")
        evidence = raw.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("purpose binding lacks hash-bound evidence")
        required_hashes = (
            "source_index_sha256",
            "points_sha256",
            "sidecar_sha256",
            "instances_sha256",
            "motion_index_sha256",
            "dense_index_sha256",
        )
        for key in required_hashes:
            digest = str(evidence.get(key, "")).lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("purpose binding {} is malformed".format(key))
        if evidence.get("source_index_sha256") != source_index_sha:
            raise ValueError("purpose binding source index hash changed")
        for key in (
            "points_file",
            "sidecar_file",
            "instances_file",
            "motion_index_file",
            "dense_index_file",
        ):
            if not str(evidence.get(key, "")).strip():
                raise ValueError("purpose binding {} is absent".format(key))
        for key in ("purpose_point_count", "target_point_count"):
            count = evidence.get(key)
            if isinstance(count, bool) or not isinstance(count, int) or count < 8:
                raise ValueError("purpose binding {} is invalid".format(key))
        source_sha = str(evidence.get("source_sha256", "")).lower()
        payload = dict(evidence)
        payload.pop("source_sha256", None)
        canonical = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if source_sha != canonical:
            raise ValueError("purpose binding canonical evidence hash changed")
        observed[(scene, prompt)] = dict(raw)
    expected_keys = {
        (scene, prompt)
        for scene in SEALED_PURPOSE_BINDINGS
        for prompt in PROMPT_PURPOSE_CATEGORY
    }
    if set(observed) != expected_keys:
        raise ValueError("purpose manifest is incomplete")
    return {
        "scene_ids": sorted(SEALED_PURPOSE_BINDINGS),
        "prompt_ids": list(PROMPT_PURPOSE_CATEGORY),
        "bindings": observed,
    }


def validate_purpose_manifest_source_index(
    value: Mapping[str, object],
    source_index_path: Path,
    teacher_source_index_sha256: str,
) -> Dict[str, object]:
    """Require manifest, Teacher summary, and disk to name one source index.

    An internally valid old manifest must never be reusable with a Teacher
    summary produced from a different dataset index.  The caller supplies the
    digest sealed by Teacher-v10.11; this function independently recomputes
    the file digest and compares all three authorities.
    """
    validate_purpose_manifest(value)
    source_index = Path(source_index_path).expanduser().resolve()
    if not source_index.is_file():
        raise FileNotFoundError(source_index)
    manifest_digest = str(value.get("source_index_sha256", "")).lower()
    teacher_digest = str(teacher_source_index_sha256).lower()
    if len(teacher_digest) != 64 or any(
        character not in "0123456789abcdef" for character in teacher_digest
    ):
        raise ValueError("Teacher source-index SHA-256 is malformed")
    disk_digest = sha256_file(source_index)
    if teacher_digest != disk_digest:
        raise ValueError("Teacher source-index SHA-256 differs from disk")
    if manifest_digest != disk_digest:
        raise ValueError("purpose manifest belongs to a different source index")
    return {
        "path": str(source_index),
        "sha256": disk_digest,
        "teacher_summary_sha256": teacher_digest,
        "purpose_manifest_sha256": manifest_digest,
        "verified": True,
    }


def canonical_source_identity(row: Mapping[str, object]) -> Tuple[str, ...]:
    """Return one nonempty recording identity, excluding scene-local names.

    ``motion_id`` and transformed-motion paths are intentionally not identity
    keys.  New High-Desk rows carry the preferred recording metadata, allowing
    cross-scene reuse to be detected even when scene-local TXT hashes differ.
    Older rows carry no such metadata, so their mandatory source-motion digest
    is the only contract-backed fallback identity.  Partial metadata is
    rejected rather than silently switching identity schemes.
    """

    digest = str(row.get(SOURCE_MOTION_HASH_KEY, "")).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("canonical source motion SHA-256 is malformed")
    values = tuple(str(row.get(name, "")).strip() for name in SOURCE_IDENTITY_KEYS)
    if all(values):
        return ("recording_metadata_v1",) + values
    if any(values):
        raise ValueError("canonical source identity metadata is partial")
    return ("source_motion_sha256_v1", digest)


def validate_source_identity_split(
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Seal identity-aligned ranks and prove global train/validation disjointness."""

    expected_panels = {
        (scene, prompt)
        for scene in SEALED_PURPOSE_BINDINGS
        for prompt in PROMPT_PURPOSE_CATEGORY
    }
    if len(records) != len(expected_panels) * 6:
        raise ValueError("source identity split must contain exactly 24 motions")
    panels: Dict[Tuple[str, str], Dict[int, Tuple[str, ...]]] = {}
    rank_by_identity: Dict[Tuple[str, ...], int] = {}
    identity_by_motion_hash: Dict[str, Tuple[str, ...]] = {}
    hash_by_scene_identity: Dict[Tuple[str, Tuple[str, ...]], str] = {}
    normalized = []
    train_identities = set()
    validation_identities = set()
    train_motion_hashes = set()
    validation_motion_hashes = set()
    for raw in records:
        scene = str(raw.get("scene_id", ""))
        prompt = str(raw.get("prompt_id", ""))
        motion = str(raw.get("motion_id", "")).strip()
        rank = raw.get("motion_rank")
        split = str(raw.get("split", ""))
        if (
            (scene, prompt) not in expected_panels
            or not motion
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
            or rank >= 6
            or split != ("train" if rank < 4 else "validation")
        ):
            raise ValueError("source identity split row metadata changed")
        identity = canonical_source_identity(raw)
        motion_hash = str(raw[SOURCE_MOTION_HASH_KEY]).strip().lower()
        panel = panels.setdefault((scene, prompt), {})
        if rank in panel or identity in panel.values():
            raise ValueError("source identity panel rank/identity is duplicated")
        panel[rank] = identity
        previous = rank_by_identity.setdefault(identity, rank)
        if previous != rank:
            raise ValueError("canonical source identity changed rank across panels")
        previous_identity = identity_by_motion_hash.setdefault(motion_hash, identity)
        if previous_identity != identity:
            raise ValueError("source motion SHA-256 is aliased by multiple identities")
        scene_identity_key = (scene, identity)
        previous_hash = hash_by_scene_identity.setdefault(
            scene_identity_key, motion_hash
        )
        if previous_hash != motion_hash:
            raise ValueError(
                "canonical source identity maps to multiple motion hashes in one scene"
            )
        (train_identities if rank < 4 else validation_identities).add(identity)
        (
            train_motion_hashes if rank < 4 else validation_motion_hashes
        ).add(motion_hash)
        normalized.append(
            {
                "scene_id": scene,
                "prompt_id": prompt,
                "motion_id": motion,
                "motion_rank": rank,
                "split": split,
                **{
                    name: str(raw.get(name, "")).strip()
                    for name in SOURCE_IDENTITY_KEYS
                },
                SOURCE_MOTION_HASH_KEY: motion_hash,
                "canonical_source_identity": list(identity),
            }
        )
    if set(panels) != expected_panels or any(
        set(panel) != set(range(6)) for panel in panels.values()
    ):
        raise ValueError("source identity split panel/rank inventory changed")
    for panel in panels.values():
        ranked = [panel[index] for index in range(6)]
        if ranked != sorted(ranked):
            raise ValueError("source identity ranks are not canonical lexicographic order")
    overlap = train_identities.intersection(validation_identities)
    if overlap:
        raise ValueError("canonical source identity leaked across train/validation")
    hash_overlap = train_motion_hashes.intersection(validation_motion_hashes)
    if hash_overlap:
        raise ValueError("source motion SHA-256 leaked across train/validation")
    normalized.sort(
        key=lambda row: (
            row["scene_id"],
            row["prompt_id"],
            row["motion_rank"],
            row["motion_id"],
        )
    )
    return {
        "identity_policy": {
            "preferred_keys": list(SOURCE_IDENTITY_KEYS),
            "legacy_fallback_key": SOURCE_MOTION_HASH_KEY,
            "partial_preferred_metadata_allowed": False,
            "scene_local_motion_id_is_identity": False,
        },
        "rank_policy": "canonical_identity_lexicographic_v1",
        "train_ranks": [0, 1, 2, 3],
        "validation_ranks": [4, 5],
        "rank_identity_alignment_exact": True,
        "source_motion_sha256_alias_free": True,
        "scene_identity_source_motion_sha256_consistent": True,
        "train_validation_source_identity_overlap": [],
        "train_validation_source_motion_sha256_overlap": [],
        "train_source_identities": [list(value) for value in sorted(train_identities)],
        "validation_source_identities": [
            list(value) for value in sorted(validation_identities)
        ],
        "train_source_motion_sha256": sorted(train_motion_hashes),
        "validation_source_motion_sha256": sorted(validation_motion_hashes),
        "records": normalized,
    }


def canonical_category_mask(
    category_ids: np.ndarray,
    numeric_to_name: Mapping[int, str],
    category_name: str,
) -> np.ndarray:
    """Build a training-only mask after explicit string-name canonicalization."""
    ids = np.asarray(category_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("category_ids must be an integer vector")
    mapping = {
        int(key): str(value).strip().lower() for key, value in numeric_to_name.items()
    }
    if len(mapping) != len(numeric_to_name):
        raise ValueError("category mapping has duplicate numeric keys")
    wanted = str(category_name).strip().lower()
    matching = [key for key, name in mapping.items() if name == wanted]
    if len(matching) != 1:
        raise ValueError("category name must map to exactly one numeric value")
    mask = ids == matching[0]
    if int(mask.sum()) < 8:
        raise ValueError("purpose category mask is absent/too small: " + wanted)
    return mask


def _pose_heading_xy_numpy(pose_value: np.ndarray) -> np.ndarray:
    """Derive one unit body-forward vector without reading another frame."""
    pose = np.asarray(pose_value, dtype=np.float32)
    if pose.shape != (22, 3) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite [22,3]")
    right_axis = (pose[2, :2] - pose[1, :2]) + (pose[17, :2] - pose[16, :2])
    norm = float(np.linalg.norm(right_axis))
    if norm < 1e-4:
        right_axis = pose[2, :2] - pose[1, :2]
        norm = float(np.linalg.norm(right_axis))
    if norm < 1e-4:
        raise ValueError("pose has no stable left/right body axis")
    right_axis = right_axis / np.float32(norm)
    forward = np.asarray((-right_axis[1], right_axis[0]), dtype=np.float32)
    if not np.isfinite(forward).all() or not np.isclose(
        np.linalg.norm(forward), 1.0, rtol=0.0, atol=1e-5
    ):
        raise AssertionError("pose heading normalization failed")
    return forward


def initial_pose_state_numpy(first_pose: np.ndarray) -> np.ndarray:
    """Legacy helper for exact v3 replay; v4 state uses the history endpoint."""
    pose = np.asarray(first_pose, dtype=np.float32)
    if pose.shape != (22, 3) or not np.isfinite(pose).all():
        raise ValueError("initial pose must be finite [22,3]")
    forward = _pose_heading_xy_numpy(pose)
    state = np.concatenate((pose[0, :2], forward)).astype(np.float32)
    if not np.isfinite(state).all() or not np.isclose(
        np.linalg.norm(state[2:]), 1.0, rtol=0.0, atol=1e-5
    ):
        raise AssertionError("initial-pose state normalization failed")
    return state


def validate_observed_history_numpy(
    observed_history: np.ndarray,
) -> Dict[str, object]:
    """Validate the mandatory raw ``[B,8,6]`` observed-prefix tensor."""
    if not isinstance(observed_history, np.ndarray):
        raise TypeError("observed_history must be a NumPy array")
    if observed_history.dtype != np.dtype(np.float32):
        raise TypeError("observed_history must be float32")
    if (
        observed_history.ndim != 3
        or observed_history.shape[0] < 1
        or observed_history.shape[1:] != (
            OBSERVED_HISTORY_FRAMES,
            OBSERVED_HISTORY_DIM,
        )
    ):
        raise ValueError("observed_history must have shape [B,8,6]")
    if not np.isfinite(observed_history).all():
        raise ValueError("observed_history contains NaN/Inf")
    heading_norm = np.linalg.norm(observed_history[..., 4:6], axis=-1)
    if not np.allclose(
        heading_norm,
        np.ones_like(heading_norm),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("every observed_history heading must be unit length")
    return {
        "shape": tuple(int(value) for value in observed_history.shape),
        "channels": OBSERVED_HISTORY_CHANNELS,
        "ordering": "oldest_to_newest",
        "finite": True,
        "unit_headings": True,
    }


def observed_history_prefix_numpy(
    joint_positions22_adm_chair_local_z_up: np.ndarray,
    target_times_s: np.ndarray,
) -> np.ndarray:
    """Reconstruct frames 0..7 as raw history, never reading suffix values.

    The returned per-motion value is ``[8,6]`` and is intended to be stacked
    into the runtime ``[B,8,6]`` tensor.  Velocity at frame zero is sealed to
    zero; every later velocity is a backward difference using only two frames
    inside the observed prefix.  A ninth source frame is required solely to
    prove that the supervised future suffix beginning at frame 8 is nonempty.
    """
    positions_value = np.asarray(
        joint_positions22_adm_chair_local_z_up, dtype=np.float32
    )
    times_value = np.asarray(target_times_s, dtype=np.float64)
    if (
        positions_value.ndim != 3
        or positions_value.shape[1:] != (22, 3)
        or positions_value.shape[0] <= FUTURE_SUFFIX_START_FRAME
    ):
        raise ValueError("motion must have shape [T,22,3] with T >= 9")
    if times_value.shape != (positions_value.shape[0],):
        raise ValueError("target_times_s must have shape [T]")

    # Deliberately slice before content validation.  The reconstruction below
    # has no data dependency on target frames 8..T-1.
    prefix = positions_value[:OBSERVED_HISTORY_FRAMES]
    prefix_times = times_value[:OBSERVED_HISTORY_FRAMES]
    if not np.isfinite(prefix).all() or not np.isfinite(prefix_times).all():
        raise ValueError("observed prefix contains NaN/Inf")
    delta_t = np.diff(prefix_times)
    if np.any(delta_t <= 0.0):
        raise ValueError("observed target times must be strictly increasing")

    pelvis_xy = prefix[:, 0, :2].astype(np.float32, copy=True)
    velocity_xy = np.zeros((OBSERVED_HISTORY_FRAMES, 2), dtype=np.float32)
    velocity_xy[1:] = (
        np.diff(pelvis_xy.astype(np.float64), axis=0) / delta_t[:, None]
    ).astype(np.float32)
    headings = np.stack(
        [_pose_heading_xy_numpy(pose) for pose in prefix], axis=0
    ).astype(np.float32)
    history = np.concatenate((pelvis_xy, velocity_xy, headings), axis=-1).astype(
        np.float32
    )
    validate_observed_history_numpy(history[None])
    return history


def history_endpoint_state_numpy(observed_history: np.ndarray) -> np.ndarray:
    """Return ``[B,4]`` state from frame 7 of a validated raw history."""
    validate_observed_history_numpy(observed_history)
    state = np.concatenate(
        (observed_history[:, -1, :2], observed_history[:, -1, 4:6]), axis=-1
    ).astype(np.float32)
    if not np.isfinite(state).all():
        raise AssertionError("history endpoint state normalization failed")
    return state


def validate_observed_prefix_suffix_split(
    observed_frame_indices: Sequence[int],
    future_suffix_start_frame: int,
    source_frame_count: int,
) -> Dict[str, object]:
    """Prove exact, disjoint observed-prefix/future-suffix frame ownership."""
    observed = tuple(observed_frame_indices)
    if observed != tuple(range(OBSERVED_HISTORY_FRAMES)):
        raise ValueError("observed frame indices must be exactly 0..7")
    if (
        isinstance(future_suffix_start_frame, bool)
        or future_suffix_start_frame != FUTURE_SUFFIX_START_FRAME
    ):
        raise ValueError("future suffix must begin at frame 8")
    if (
        isinstance(source_frame_count, bool)
        or not isinstance(source_frame_count, int)
        or source_frame_count <= FUTURE_SUFFIX_START_FRAME
    ):
        raise ValueError("source motion must contain a nonempty future suffix")
    if max(observed) >= future_suffix_start_frame:
        raise AssertionError("observed prefix overlaps the future suffix")
    return {
        "observed_frame_indices": observed,
        "future_suffix_start_frame": future_suffix_start_frame,
        "future_suffix_frame_count": source_frame_count - future_suffix_start_frame,
        "disjoint": True,
    }


def fuse_two_branch_logits_numpy(
    relation_raw_logits: np.ndarray,
    history_raw_logits: np.ndarray,
    *,
    relation_coefficient: float = RELATION_BRANCH_COEFFICIENT,
    history_coefficient: float = HISTORY_BRANCH_COEFFICIENT,
) -> Dict[str, np.ndarray]:
    """Apply the exact image-(b) ``0.6*w_R + 0.4*w_H`` fusion.

    Each branch first maps its independent expert-mixture logit through the
    sigmoid-equivalent ``(tanh(logit/2)+1)/2``.  Coefficients are arguments
    only so validators can prove that attempts to alter them fail closed.
    """
    for name, actual, sealed in (
        ("relation_coefficient", relation_coefficient, RELATION_BRANCH_COEFFICIENT),
        ("history_coefficient", history_coefficient, HISTORY_BRANCH_COEFFICIENT),
    ):
        try:
            numeric = float(actual)
        except (TypeError, ValueError) as exc:
            raise TypeError(name + " must be a finite float") from exc
        if not math.isfinite(numeric):
            raise ValueError(name + " must be finite")
        if numeric != float(sealed):
            raise ValueError(
                "{} changed from sealed value {} to {}".format(
                    name, sealed, numeric
                )
            )
    if float(relation_coefficient) + float(history_coefficient) != 1.0:
        raise AssertionError("sealed fusion coefficients no longer sum to one")

    relation = np.asarray(relation_raw_logits, dtype=np.float32)
    history = np.asarray(history_raw_logits, dtype=np.float32)
    if relation.ndim != 2 or history.shape != relation.shape:
        raise ValueError("branch logits must have one shared shape [B,N]")
    if not np.isfinite(relation).all() or not np.isfinite(history).all():
        raise ValueError("branch logits contain NaN/Inf")
    relation_signed = np.tanh(relation / np.float32(2.0)).astype(np.float32)
    history_signed = np.tanh(history / np.float32(2.0)).astype(np.float32)
    relation_weight = (
        (relation_signed + np.float32(1.0)) / np.float32(2.0)
    ).astype(np.float32)
    history_weight = (
        (history_signed + np.float32(1.0)) / np.float32(2.0)
    ).astype(np.float32)
    fused_weight = (
        np.float32(RELATION_BRANCH_COEFFICIENT) * relation_weight
        + np.float32(HISTORY_BRANCH_COEFFICIENT) * history_weight
    ).astype(np.float32)
    fused_signed = (
        np.float32(2.0) * fused_weight - np.float32(1.0)
    ).astype(np.float32)
    return {
        "relation_weight": relation_weight,
        "history_weight": history_weight,
        "relation_signed_correction": relation_signed,
        "history_signed_correction": history_signed,
        "fused_weight": fused_weight,
        "fused_signed_correction": fused_signed,
    }


def bounded_two_branch_map_numpy(
    teacher: np.ndarray,
    relation_raw_logits: np.ndarray,
    history_raw_logits: np.ndarray,
    support: np.ndarray,
    *,
    relation_coefficient: float = RELATION_BRANCH_COEFFICIENT,
    history_coefficient: float = HISTORY_BRANCH_COEFFICIENT,
    alpha_up: float = ALPHA_UP,
    alpha_down: float = ALPHA_DOWN,
) -> Dict[str, np.ndarray]:
    """Independently replay fixed two-branch fusion plus safe residual map."""
    fused = fuse_two_branch_logits_numpy(
        relation_raw_logits,
        history_raw_logits,
        relation_coefficient=relation_coefficient,
        history_coefficient=history_coefficient,
    )
    epsilon = np.float32(np.finfo(np.float32).eps)
    clipped_signed = np.clip(
        fused["fused_signed_correction"],
        np.float32(-1.0) + epsilon,
        np.float32(1.0) - epsilon,
    ).astype(np.float32)
    fused_raw_logits = (
        np.float32(2.0) * np.arctanh(clipped_signed)
    ).astype(np.float32)
    bounded = bounded_map_numpy(
        teacher,
        fused_raw_logits,
        support,
        alpha_up=alpha_up,
        alpha_down=alpha_down,
    )
    return {
        **bounded,
        **fused,
        "fused_raw_correction_logits": fused_raw_logits,
    }


def bounded_map_numpy(
    teacher: np.ndarray,
    raw_logits: np.ndarray,
    support: np.ndarray,
    alpha_up: float = ALPHA_UP,
    alpha_down: float = ALPHA_DOWN,
) -> Dict[str, np.ndarray]:
    """Independent NumPy implementation used by validators."""
    for name, actual, sealed in (
        ("alpha_up", alpha_up, ALPHA_UP),
        ("alpha_down", alpha_down, ALPHA_DOWN),
    ):
        try:
            numeric = float(actual)
        except (TypeError, ValueError) as exc:
            raise TypeError(name + " must be a finite float") from exc
        if not math.isfinite(numeric):
            raise ValueError(name + " must be finite")
        if numeric != float(sealed):
            raise ValueError(
                "{} changed from sealed value {} to {}".format(name, sealed, numeric)
            )
    base = np.asarray(teacher, dtype=np.float32)
    logits = np.asarray(raw_logits, dtype=np.float32)
    candidate = np.asarray(support, dtype=np.float32)
    if base.ndim != 3 or base.shape[-1] != CONTACT_DIM:
        raise ValueError("teacher must have shape [B,N,6]")
    if logits.shape != base.shape[:2] or candidate.shape != base.shape[:2]:
        raise ValueError("bounded-map axes do not align")
    if not all(np.isfinite(value).all() for value in (base, logits, candidate)):
        raise ValueError("bounded-map input is non-finite")
    if np.any(base < 0.0) or np.any(base > 1.0):
        raise ValueError("teacher lies outside [0,1]")
    if np.any(candidate < 0.0) or np.any(candidate > 1.0):
        raise ValueError("support lies outside [0,1]")
    q = np.tanh(logits / np.float32(2.0)).astype(np.float32)
    weight = ((q + np.float32(1.0)) / np.float32(2.0)).astype(np.float32)
    signed = q[..., None]
    supported = candidate[..., None]
    up = base + np.float32(alpha_up) * supported * signed * (np.float32(1.0) - base)
    down = base + np.float32(alpha_down) * supported * signed * base
    corrected = np.where(signed >= 0.0, up, down).astype(np.float32)
    lower = ((np.float32(1.0) - np.float32(alpha_down) * supported) * base).astype(
        np.float32
    )
    upper = (base + np.float32(alpha_up) * supported * (np.float32(1.0) - base)).astype(
        np.float32
    )
    corrected = np.maximum(np.minimum(corrected, upper), lower).astype(np.float32)
    corrected = np.clip(corrected, np.float32(0.0), np.float32(1.0)).astype(np.float32)
    return {
        "mapstar": corrected,
        "weight": weight,
        "signed_correction": q,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def candidate_support_numpy(
    teacher_affordance: np.ndarray,
    legacy_terminal_iiw: np.ndarray,
    candidate_probability: np.ndarray,
    purpose_probability: np.ndarray,
    *,
    support_floor: float = SEALED_SUPPORT_FLOOR,
    support_ceiling: float = SEALED_SUPPORT_CEILING,
) -> np.ndarray:
    """Recompute inference support from its four independently saved inputs.

    This mirrors ``candidate_support_from_predictions`` but intentionally uses
    only NumPy and probabilities saved before the support combination.  The
    saved ``candidate_support`` array is therefore evidence to verify, never an
    authority that can make its own bounded-map formula appear valid.
    """
    for name, value in (
        ("teacher_affordance", teacher_affordance),
        ("legacy_terminal_iiw", legacy_terminal_iiw),
        ("candidate_probability", candidate_probability),
        ("purpose_probability", purpose_probability),
    ):
        if not isinstance(value, np.ndarray):
            raise TypeError(name + " must be a NumPy array")
        if value.dtype != np.dtype(np.float32):
            raise TypeError(name + " must be float32")
        if not np.isfinite(value).all():
            raise ValueError(name + " contains NaN/Inf")
    teacher = teacher_affordance
    legacy = legacy_terminal_iiw
    candidate = candidate_probability
    purpose = purpose_probability
    if teacher.ndim != 3 or teacher.shape[-1] != CONTACT_DIM:
        raise ValueError("teacher_affordance must have shape [B,N,6]")
    if legacy.shape != teacher.shape:
        raise ValueError("legacy_terminal_iiw shape mismatch")
    if candidate.shape != teacher.shape[:2] or purpose.shape != teacher.shape[:2]:
        raise ValueError("candidate/purpose probability axes do not align")
    for name, value in (
        ("teacher_affordance", teacher),
        ("legacy_terminal_iiw", legacy),
        ("candidate_probability", candidate),
        ("purpose_probability", purpose),
    ):
        if float(value.min()) < 0.0 or float(value.max()) > 1.0:
            raise ValueError(name + " lies outside [0,1]")
    if float(support_floor) != SEALED_SUPPORT_FLOOR:
        raise ValueError("support_floor changed from sealed value")
    if float(support_ceiling) != SEALED_SUPPORT_CEILING:
        raise ValueError("support_ceiling changed from sealed value")

    floor = np.float32(SEALED_SUPPORT_FLOOR)
    ceiling = np.float32(SEALED_SUPPORT_CEILING)
    one = np.float32(1.0)
    teacher_scalar = teacher.max(axis=-1)
    teacher_support = np.clip(
        (teacher_scalar - floor) / (ceiling - floor),
        np.float32(0.0),
        one,
    ).astype(np.float32)
    legacy_support = np.clip(
        legacy.max(axis=-1), np.float32(0.0), one
    ).astype(np.float32)
    frozen_union = (one - (one - teacher_support) * (one - legacy_support)).astype(
        np.float32
    )
    candidate_gate = (floor + (one - floor) * candidate).astype(np.float32)
    gated_frozen = (frozen_union * candidate_gate).astype(np.float32)
    evidence_union = (one - (one - candidate) * (one - gated_frozen)).astype(
        np.float32
    )
    return np.clip(
        evidence_union * (one - purpose), np.float32(0.0), one
    ).astype(np.float32)


def strict_status(checks: Mapping[str, object]) -> Tuple[str, Tuple[str, ...]]:
    """Top-level PASS is impossible unless every quality check passes."""
    if not isinstance(checks, Mapping):
        raise TypeError("checks must be a mapping")
    missing = tuple(name for name in MANDATORY_CHECKS if name not in checks)
    if missing:
        raise ValueError("mandatory checks missing: " + repr(missing))
    failed = tuple(name for name in MANDATORY_CHECKS if checks[name] is not True)
    return ("RESEARCH_ONLY_PASS" if not failed else "RESEARCH_ONLY_FAIL"), failed


def instantaneous_status(
    checks: Mapping[str, object],
) -> Tuple[str, Tuple[str, ...]]:
    """Evaluate model/data gates without the run-level stable-streak gate."""
    if not isinstance(checks, Mapping):
        raise TypeError("instantaneous checks must be a mapping")
    missing = tuple(name for name in INSTANTANEOUS_CHECKS if name not in checks)
    if missing:
        raise ValueError("instantaneous checks missing: " + repr(missing))
    failed = tuple(name for name in INSTANTANEOUS_CHECKS if checks[name] is not True)
    return ("RESEARCH_ONLY_PASS" if not failed else "RESEARCH_ONLY_FAIL"), failed


def validate_stable_pass_streak(
    history: Sequence[object], finite_search: Mapping[str, object]
) -> Dict[str, object]:
    """Recompute the eligible consecutive-PASS streak from sealed history.

    PASS observations before ``minimum_steps`` never increment the streak.  A
    stable run must stop at the first qualifying streak end, and that end must
    be the selected ``best_step``.  Recorded booleans/counters are checked but
    never used as the source of truth.
    """

    if not isinstance(finite_search, Mapping):
        raise TypeError("finite_search must be a mapping")
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise TypeError("history must be a sequence")

    def positive_int(name: str, *, allow_zero: bool = False) -> int:
        value = finite_search.get(name)
        lower = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < lower:
            raise ValueError(name + " must be an integer >= " + str(lower))
        return value

    maximum_steps = positive_int("maximum_steps")
    completed_steps = positive_int("completed_steps")
    minimum_steps = positive_int("minimum_steps")
    eval_every = positive_int("eval_every")
    required = positive_int("required_consecutive_passes")
    best_step = positive_int("best_step", allow_zero=True)
    sealed_policy = {
        "maximum_steps": SEALED_MAXIMUM_STEPS,
        "minimum_steps": SEALED_MINIMUM_STEPS,
        "eval_every": SEALED_EVAL_EVERY,
        "required_consecutive_passes": SEALED_REQUIRED_CONSECUTIVE_PASSES,
    }
    for name, expected in sealed_policy.items():
        if finite_search.get(name) != expected:
            raise ValueError(
                "finite_search {} must equal sealed value {}".format(name, expected)
            )
    if finite_search.get("contrast_every") != SEALED_CONTRAST_EVERY:
        raise ValueError(
            "finite_search contrast_every must equal sealed value "
            + str(SEALED_CONTRAST_EVERY)
        )
    if finite_search.get("relation_geometry_contrast_schedule") != list(
        SEALED_RELATION_GEOMETRY_CONTRAST_SCHEDULE
    ):
        raise ValueError("finite_search relation-geometry contrast schedule changed")
    if minimum_steps > maximum_steps or completed_steps > maximum_steps:
        raise ValueError("finite_search step bounds are inconsistent")
    if finite_search.get("infinite_retry") is not False:
        raise ValueError("finite_search must explicitly disable infinite retry")

    expected_steps = list(range(eval_every, completed_steps + 1, eval_every))
    if completed_steps == maximum_steps and (
        not expected_steps or expected_steps[-1] != completed_steps
    ):
        expected_steps.append(completed_steps)
    if not expected_steps:
        raise ValueError("finite search contains no completed evaluation")

    eval_steps = []
    instantaneous_passes = []
    eligible_streaks = []
    streak = 0
    first_qualifying_index = None
    for index, raw_entry in enumerate(history):
        if not isinstance(raw_entry, Mapping):
            raise TypeError("history entries must be mappings")
        step = raw_entry.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("history step must be a positive integer")
        eval_steps.append(step)

        recorded_checks = raw_entry.get("instantaneous_checks")
        if not isinstance(recorded_checks, Mapping):
            raise TypeError("history instantaneous_checks must be a mapping")
        if set(recorded_checks) != set(INSTANTANEOUS_CHECKS):
            raise ValueError("history instantaneous check inventory changed")
        status, failures = instantaneous_status(recorded_checks)
        passed = not failures
        eligible = step >= minimum_steps
        streak = streak + 1 if eligible and passed else 0
        instantaneous_passes.append(passed)
        eligible_streaks.append(streak)

        if raw_entry.get("status") != status:
            raise ValueError("history status disagrees with instantaneous checks")
        if raw_entry.get("failed_checks") != list(failures):
            raise ValueError("history failed_checks disagree with checks")
        if raw_entry.get("instantaneous_pass") is not passed:
            raise ValueError("history instantaneous_pass disagrees with checks")
        if raw_entry.get("eligible_for_streak") is not eligible:
            raise ValueError("history eligibility disagrees with minimum_steps")
        recorded_streak = raw_entry.get("eligible_pass_streak")
        if (
            isinstance(recorded_streak, bool)
            or not isinstance(recorded_streak, int)
            or recorded_streak != streak
        ):
            raise ValueError("history eligible PASS streak counter changed")
        if streak >= required and first_qualifying_index is None:
            first_qualifying_index = index

    if eval_steps != expected_steps:
        raise ValueError("history evaluation steps disagree with finite search")
    if best_step not in eval_steps:
        raise ValueError("best_step was not independently evaluated")

    stable = first_qualifying_index is not None
    qualifying_steps = []
    if stable:
        end_index = int(first_qualifying_index)
        start_index = end_index - required + 1
        qualifying_steps = eval_steps[start_index : end_index + 1]
        qualifying_end = qualifying_steps[-1]
        if best_step != qualifying_end:
            raise ValueError("selected best_step is not the qualifying streak end")
        if completed_steps != qualifying_end or end_index != len(eval_steps) - 1:
            raise ValueError("finite search continued past first qualifying streak")

    if finite_search.get(STABLE_PASS_CHECK) is not stable:
        raise ValueError("finite_search stable PASS claim disagrees with history")
    if finite_search.get("qualifying_streak_steps") != qualifying_steps:
        raise ValueError("finite_search qualifying streak steps changed")

    return {
        STABLE_PASS_CHECK: stable,
        "eval_steps": tuple(eval_steps),
        "instantaneous_passes": tuple(instantaneous_passes),
        "eligible_pass_streaks": tuple(eligible_streaks),
        "qualifying_streak_steps": tuple(qualifying_steps),
    }


def checkpoint_authorized(summary: Mapping[str, object]) -> bool:
    """Return whether a research checkpoint may be emitted.

    This does not authorize Teacher promotion or production deployment.
    """
    if not isinstance(summary, Mapping):
        return False
    try:
        status, failed = strict_status(summary.get("checks", {}))
    except (TypeError, ValueError):
        return False
    schema = summary.get("schema")
    if schema == SCHEMA + "_training_summary_v1":
        try:
            streak = validate_stable_pass_streak(
                summary.get("history", ()), summary.get("finite_search", {})
            )
        except (TypeError, ValueError):
            return False
        stable_authorized = streak[STABLE_PASS_CHECK] is True
    elif schema == SCHEMA + "_checkpoint_v1":
        # Checkpoint structure/tensors are validated separately by the strict
        # loader.  Keep inference compatible while refusing diagnostic
        # candidates and avoiding a second, unaudited history copy in weights.
        stable_authorized = summary.get("checkpoint_authorized") is True
    else:
        return False
    return bool(
        status == "RESEARCH_ONLY_PASS"
        and not failed
        and stable_authorized
        and summary.get("held_out_scene_generalization") is False
        and summary.get("production_authorized") is False
        and summary.get("teacher_promotion_authorized") is False
    )


__all__ = (
    "ALPHA_DOWN",
    "ALPHA_UP",
    "BRANCH_EXPERT_COUNT",
    "CANONICAL_CLIP_MAX_LENGTH",
    "CANONICAL_CLIP_VERSION",
    "CANONICAL_PROMPT_BINDING_SCHEMA",
    "CANONICAL_PROMPT_IDS",
    "CANONICAL_PROMPT_TEXTS",
    "CONTACT_DIM",
    "FORMAT_VERSION",
    "FORWARD_INPUTS",
    "FUTURE_SUFFIX_START_FRAME",
    "FAMILY_ABLATION_LOSS_TOLERANCE",
    "FAMILY_ABLATION_MAP_MAE_MIN",
    "IDENTITY_WEIGHT",
    "INSTANTANEOUS_CHECKS",
    "MANDATORY_CHECKS",
    "HISTORY_BRANCH_COEFFICIENT",
    "HISTORY_FEATURE_NAMES",
    "OBSERVED_HISTORY_CHANNELS",
    "OBSERVED_HISTORY_CONTRACT",
    "OBSERVED_HISTORY_DIM",
    "OBSERVED_HISTORY_FRAMES",
    "POINT_COUNT",
    "PROMPT_EXPERT",
    "PROMPT_PURPOSE_CATEGORY",
    "RELATION_GEOMETRY_ABLATION_FAMILIES",
    "RELATION_FEATURE_NAMES",
    "RELATION_BRANCH_COEFFICIENT",
    "RELATION_SCOPE_POLICY",
    "SCHEMA",
    "SEALED_PURPOSE_BINDINGS",
    "SEALED_SUPPORT_CEILING",
    "SEALED_SUPPORT_FLOOR",
    "SOURCE_IDENTITY_KEYS",
    "SOURCE_MOTION_HASH_KEY",
    "SUPPORTED_RELATION_PROMPT_IDS",
    "STATUS",
    "STABLE_PASS_CHECK",
    "SEALED_EVAL_EVERY",
    "SEALED_CONTRAST_EVERY",
    "SEALED_MAXIMUM_STEPS",
    "SEALED_MINIMUM_STEPS",
    "SEALED_REQUIRED_CONSECUTIVE_PASSES",
    "SEALED_RELATION_GEOMETRY_CONTRAST_SCHEDULE",
    "WEIGHT_RANGE",
    "assert_forward_input_contract",
    "bounded_map_numpy",
    "bounded_two_branch_map_numpy",
    "candidate_support_numpy",
    "canonical_source_identity",
    "canonical_category_mask",
    "checkpoint_authorized",
    "instantaneous_status",
    "history_endpoint_state_numpy",
    "initial_pose_state_numpy",
    "observed_history_prefix_numpy",
    "sha256_file",
    "strict_status",
    "validate_purpose_manifest",
    "validate_purpose_manifest_source_index",
    "validate_observed_history_numpy",
    "validate_observed_prefix_suffix_split",
    "validate_source_identity_split",
    "validate_stable_pass_streak",
    "fuse_two_branch_logits_numpy",
    "TWO_BRANCH_MOE_METADATA",
)
