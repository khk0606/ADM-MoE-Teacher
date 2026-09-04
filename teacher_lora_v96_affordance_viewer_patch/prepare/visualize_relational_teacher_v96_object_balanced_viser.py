#!/usr/bin/env python3
"""Read-only six-panel viewer for saved Teacher-v9.6 one-step maps."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from relational_teacher_v96_affordance_viewer_common import (
    CHANNEL_ORDER,
    OBJECT_LABELS,
    OBJECT_ORDER,
    affordance_colors,
    build_xy_interpolation_plan,
    interpolate_xy_heatmap,
    rgba_heatmap,
    scalar_channel,
    signed_difference_colors,
    topk_change_masks,
)


SCHEMA = "relational_teacher_v96_object_balanced_consensus_v1"
POINT_COUNT = 8192
AUDIT_NAMES = ("audit_t010", "audit_t100", "audit_t225", "audit_t490")
DEFAULT_CANDIDATE = "chair_pair_priority_radius_0p001"
EXPECTED_ARRAY_KEYS = {
    "xyz",
    "verified_object_mask",
    "verified_positive_mask",
    "unknown_sittable_mask",
    "explicit_negative_mask",
    "instance_targets",
    "base",
    "candidates",
    "base_normalized",
    "candidates_normalized",
    "candidate_names",
    "v5_gt_normalized",
    "base_v5_normalized",
    "candidate_v5_normalized",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Teacher-v9.6 one-step affordance-map audit."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--initial-candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--heatmap-resolution", type=int, default=128)
    parser.add_argument("--heatmap-neighbors", type=int, default=8)
    return parser.parse_args()


def read_json(path: Path) -> Mapping[str, object]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON object: " + str(path))
    return value


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_under(root: Path, raw: object) -> Path:
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    result = candidate.resolve()
    if result != root and root not in result.parents:
        raise ValueError("path escapes bound root: " + str(raw))
    return result


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def rgb_u8(points: np.ndarray) -> np.ndarray:
    rgb = np.asarray(points[:, 3:6], dtype=np.float32)
    if not np.isfinite(rgb).all() or float(rgb.min()) < 0.0:
        raise ValueError("source RGB is invalid")
    if float(rgb.max()) <= 1.0 + 1e-6:
        rgb *= np.float32(255.0)
    elif float(rgb.max()) > 255.0 + 1e-4:
        raise ValueError("source RGB exceeds 255")
    return np.rint(rgb).clip(0, 255).astype(np.uint8)


def _validate_report(report_file: Path) -> tuple[Mapping[str, object], Path]:
    report = read_json(report_file)
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "FAIL"
        or report.get("selected_candidate") is not None
        or report.get("train_scene") != "room_0101"
        or report.get("serialized_model_state") is not False
        or report.get("heldout_train_arrays_read") is not False
        or report.get("development_arrays_read") is not False
        or report.get("paper_test_access") is not False
        or report.get("failed_checks")
        != ["at_least_one_object_balanced_candidate_admissible"]
    ):
        raise ValueError("viewer rejected the Teacher-v9.6 report authority")
    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 16:
        raise ValueError("Teacher-v9.6 candidate inventory changed")
    names = [str(row.get("name")) for row in candidates]
    if len(set(names)) != 16 or DEFAULT_CANDIDATE not in names:
        raise ValueError("Teacher-v9.6 candidate names changed")
    panels = report.get("audit_panels")
    if not isinstance(panels, list) or tuple(str(row.get("name")) for row in panels) != AUDIT_NAMES:
        raise ValueError("Teacher-v9.6 audit panels changed")
    paths = report.get("paths")
    hashes = report.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Teacher-v9.6 path binding is absent")
    maps_file = Path(str(paths.get("response_maps", ""))).expanduser().resolve()
    if not maps_file.is_file() or sha256_file(maps_file) != hashes.get("response_maps"):
        raise ValueError("Teacher-v9.6 response-map hash changed")
    return report, maps_file


def _validate_arrays(arrays: Mapping[str, np.ndarray], report: Mapping[str, object]) -> None:
    if set(arrays) != EXPECTED_ARRAY_KEYS:
        raise ValueError("Teacher-v9.6 response-map array keys changed")
    expected = {
        "xyz": (POINT_COUNT, 3),
        "verified_object_mask": (3, POINT_COUNT),
        "verified_positive_mask": (POINT_COUNT,),
        "unknown_sittable_mask": (POINT_COUNT,),
        "explicit_negative_mask": (POINT_COUNT,),
        "instance_targets": (3, POINT_COUNT, 6),
        "base": (4, 2, POINT_COUNT, 6),
        "candidates": (16, 4, 2, POINT_COUNT, 6),
        "candidate_names": (16,),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError("{} shape changed: {}".format(name, arrays[name].shape))
    for name in ("xyz", "instance_targets", "base", "candidates"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("instance_targets", "base", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")
    for name in (
        "verified_object_mask",
        "verified_positive_mask",
        "unknown_sittable_mask",
        "explicit_negative_mask",
    ):
        if arrays[name].dtype != np.bool_:
            raise ValueError(name + " is not boolean")
    masks = arrays["verified_object_mask"]
    if np.any(masks.sum(axis=0) > 1) or np.any(masks.sum(axis=1) <= 0):
        raise ValueError("verified object masks overlap or are empty")
    names = tuple(str(value) for value in arrays["candidate_names"].tolist())
    report_names = tuple(str(row["name"]) for row in report["candidates"])
    if names != report_names:
        raise ValueError("candidate array/report order changed")


def _load_scene_rgb(report: Mapping[str, object], xyz: np.ndarray) -> np.ndarray:
    paths = report["paths"]
    hashes = report["path_sha256"]
    source_index = Path(str(paths["source_dataset_index"])).expanduser().resolve()
    if not source_index.is_file() or sha256_file(source_index) != hashes["source_dataset_index"]:
        raise ValueError("bound source dataset index changed")
    source = read_json(source_index)
    records = source.get("scenes")
    if not isinstance(records, list):
        raise ValueError("source scene inventory is absent")
    matches = [row for row in records if str(row.get("scene_id")) == "room_0101"]
    if len(matches) != 1:
        raise ValueError("room_0101 source record changed")
    record = matches[0]
    root = source_index.parent.resolve()
    points_file = resolve_under(root, record["points_file"])
    if sha256_file(points_file) != str(record["points_sha256"]):
        raise ValueError("room_0101 source point-cloud hash changed")
    payload = load_npz(points_file)
    if "points" not in payload:
        raise ValueError("room_0101 source points are absent")
    points = np.asarray(payload["points"], dtype=np.float32)
    if points.shape != (POINT_COUNT, 6) or not np.isfinite(points).all():
        raise ValueError("room_0101 source points changed")
    if not np.allclose(points[:, :3], xyz, rtol=0.0, atol=1e-6):
        raise ValueError("room_0101 source/map point order differs")
    return rgb_u8(points)


def _blend(map_colors: np.ndarray, rgb: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.6))
    return np.rint((1.0 - amount) * map_colors + amount * rgb).clip(0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    report_file = args.report.expanduser().resolve()
    report, maps_file = _validate_report(report_file)
    arrays = load_npz(maps_file)
    _validate_arrays(arrays, report)
    rgb = _load_scene_rgb(report, np.asarray(arrays["xyz"], dtype=np.float32))
    candidate_names = tuple(str(value) for value in arrays["candidate_names"].tolist())
    if args.initial_candidate not in candidate_names:
        raise ValueError("initial candidate is absent: " + args.initial_candidate)
    row_by_name = {str(row["name"]): row for row in report["candidates"]}

    import viser

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label="Teacher-v9.6 one-step affordance audit",
    )
    server.gui.add_markdown(
        "## Teacher-v9.6 saved-map audit\n"
        "**저장된 room_0101 진단 map을 읽기만 합니다. 재학습·재추론 없음.**  \n"
        "위: `Input RGB → All-sittable GT → Frozen Base`  \n"
        "아래: `Candidate → Candidate − Base → |Candidate − GT|`  \n"
        "⚠️ 이것은 500-step 최종 생성물이 아니라, 고정 timestep에서의 **one-step x-start 예측**입니다.  \n"
        "⚠️ v9.6은 선택 후보 없이 끝난 **FAIL 진단 artifact**이며 checkpoint도 저장되지 않았습니다.  \n"
        "연속 면은 원본 8192개 값을 변경하지 않는 표시 전용 XY 보간입니다."
    )
    candidate = server.gui.add_dropdown(
        "Candidate",
        options=candidate_names,
        initial_value=args.initial_candidate,
    )
    audit = server.gui.add_dropdown(
        "Audit timestep",
        options=AUDIT_NAMES,
        initial_value="audit_t010",
    )
    prompt = server.gui.add_dropdown(
        "Prompt",
        options=("watch", "write"),
        initial_value="watch",
    )
    focus = server.gui.add_dropdown(
        "Object focus",
        options=("all_verified",) + OBJECT_ORDER,
        initial_value="all_verified",
    )
    channel = server.gui.add_dropdown(
        "Affordance channel",
        options=("any_joint",) + CHANNEL_ORDER,
        initial_value="any_joint",
    )
    render_mode = server.gui.add_dropdown(
        "Map rendering",
        options=("continuous heatmap + points", "continuous heatmap only", "points only"),
        initial_value="continuous heatmap + points",
    )
    rgb_blend = server.gui.add_slider(
        "RGB blend", min=0.0, max=0.60, step=0.05, initial_value=0.10
    )
    heatmap_opacity = server.gui.add_slider(
        "Continuous heatmap opacity", min=0.10, max=1.00, step=0.05, initial_value=0.85
    )
    point_size = server.gui.add_slider(
        "Point size", min=0.008, max=0.060, step=0.002, initial_value=0.020
    )
    status = server.gui.add_markdown("")
    lock = threading.Lock()

    xyz = np.asarray(arrays["xyz"], dtype=np.float32)
    center_xy = 0.5 * (xyz[:, :2].min(axis=0) + xyz[:, :2].max(axis=0))
    local_xyz = xyz.copy()
    local_xyz[:, :2] -= center_xy
    plan = build_xy_interpolation_plan(
        local_xyz,
        resolution=args.heatmap_resolution,
        neighbors=args.heatmap_neighbors,
    )
    width = float(np.ptp(local_xyz[:, 0]))
    depth = float(np.ptp(local_xyz[:, 1]))
    spacing_x = max(width + 1.0, 4.0)
    spacing_y = max(depth + 1.0, 4.0)
    offsets = (
        np.asarray((-spacing_x, 0.5 * spacing_y, 0.0), dtype=np.float32),
        np.asarray((0.0, 0.5 * spacing_y, 0.0), dtype=np.float32),
        np.asarray((spacing_x, 0.5 * spacing_y, 0.0), dtype=np.float32),
        np.asarray((-spacing_x, -0.5 * spacing_y, 0.0), dtype=np.float32),
        np.asarray((0.0, -0.5 * spacing_y, 0.0), dtype=np.float32),
        np.asarray((spacing_x, -0.5 * spacing_y, 0.0), dtype=np.float32),
    )
    object_masks = np.asarray(arrays["verified_object_mask"], dtype=bool)
    instance_targets = np.asarray(arrays["instance_targets"], dtype=np.float32)
    all_gt = np.maximum.reduce(instance_targets)
    unknown = np.asarray(arrays["unknown_sittable_mask"], dtype=bool)
    negative = np.asarray(arrays["explicit_negative_mask"], dtype=bool)

    def render() -> None:
        with lock:
            candidate_name = str(candidate.value)
            candidate_index = candidate_names.index(candidate_name)
            panel_index = AUDIT_NAMES.index(str(audit.value))
            prompt_index = 0 if str(prompt.value) == "watch" else 1
            base = np.asarray(arrays["base"][panel_index, prompt_index], dtype=np.float32)
            current = np.asarray(
                arrays["candidates"][candidate_index, panel_index, prompt_index],
                dtype=np.float32,
            )
            focus_name = str(focus.value)
            if focus_name == "all_verified":
                gt = all_gt
                focus_mask = object_masks.any(axis=0)
                metric_name = None
            else:
                slot = OBJECT_ORDER.index(focus_name)
                gt = instance_targets[slot]
                focus_mask = object_masks[slot]
                metric_name = focus_name
            selected_channel = str(channel.value)
            base_scalar = scalar_channel(base, selected_channel)
            current_scalar = scalar_channel(current, selected_channel)
            gt_scalar = scalar_channel(gt, selected_channel)
            delta = current_scalar - base_scalar
            if metric_name is None:
                absolute_error = np.abs(current_scalar - gt_scalar)
                error_label = "|Candidate − GT|"
            else:
                absolute_error = np.zeros_like(current_scalar)
                absolute_error[focus_mask] = np.abs(
                    current_scalar[focus_mask] - gt_scalar[focus_mask]
                )
                error_label = "|Candidate − GT| (focus object)"
            difference_limit = max(float(np.max(np.abs(delta))), 1e-8)
            maps = (None, gt_scalar, base_scalar, current_scalar, delta, absolute_error)
            names = (
                "Input 3D Scene (RGB)",
                "All-sittable GT" if metric_name is None else OBJECT_LABELS[metric_name] + " GT",
                "Frozen Base (one-step)",
                "Candidate (one-step)",
                "Candidate − Base",
                error_label,
            )
            for panel, (name, values, offset) in enumerate(zip(names, maps, offsets)):
                points = local_xyz + offset
                if panel == 0:
                    colors = rgb
                elif panel == 4:
                    colors = signed_difference_colors(values, difference_limit)
                else:
                    colors = _blend(affordance_colors(values), rgb, float(rgb_blend.value))
                points_visible = panel == 0 or str(render_mode.value) != "continuous heatmap only"
                server.scene.add_point_cloud(
                    f"/panel_{panel}/points",
                    points=points,
                    colors=colors,
                    point_size=float(point_size.value),
                    point_shape="circle",
                    precision="float32",
                    visible=points_visible,
                )
                surface_visible = panel != 0 and str(render_mode.value) != "points only"
                if panel != 0:
                    raster = interpolate_xy_heatmap(values, plan)
                    if panel == 4:
                        heat_colors = signed_difference_colors(raster, difference_limit)
                    else:
                        heat_colors = affordance_colors(raster)
                    xy_min = np.asarray(plan["xy_min"], dtype=np.float32)
                    xy_max = np.asarray(plan["xy_max"], dtype=np.float32)
                    position = offset + np.asarray(
                        (
                            0.5 * float(xy_min[0] + xy_max[0]),
                            0.5 * float(xy_min[1] + xy_max[1]),
                            float(np.asarray(plan["floor_z"]).item()) - 0.015,
                        ),
                        dtype=np.float32,
                    )
                    server.scene.add_image(
                        f"/panel_{panel}/heatmap",
                        image=rgba_heatmap(heat_colors, float(heatmap_opacity.value)),
                        render_width=float(xy_max[0] - xy_min[0]),
                        render_height=float(xy_max[1] - xy_min[1]),
                        position=position,
                        visible=surface_visible,
                        cast_shadow=False,
                        receive_shadow=False,
                    )
                highlight_points = points[focus_mask]
                server.scene.add_point_cloud(
                    f"/panel_{panel}/focus",
                    points=highlight_points,
                    colors=np.repeat(
                        np.asarray((35, 255, 90), dtype=np.uint8)[None],
                        highlight_points.shape[0],
                        axis=0,
                    ),
                    point_size=min(float(point_size.value) * 1.55, 0.09),
                    point_shape="circle",
                    precision="float32",
                    visible=bool(highlight_points.shape[0]),
                )
                server.scene.add_label(
                    f"/panel_{panel}/label",
                    text=name,
                    position=offset
                    + np.asarray((0.0, 0.0, float(local_xyz[:, 2].max()) + 0.55), dtype=np.float32),
                )

            row = row_by_name[candidate_name]
            base_row = row["base_rows"][panel_index]
            candidate_row = row["candidate_rows"][panel_index]
            if metric_name is None:
                topk_text = "Object focus를 Bed/Normal Chair/High Chair로 바꾸면 Top-k 교체점을 표시합니다."
                left_count = entered_count = 0
                for panel in (2, 3, 4):
                    server.scene.add_point_cloud(
                        f"/panel_{panel}/topk_left",
                        points=(local_xyz + offsets[panel])[:1],
                        colors=np.asarray([[255, 30, 60]], dtype=np.uint8),
                        point_size=0.05,
                        visible=False,
                    )
                    server.scene.add_point_cloud(
                        f"/panel_{panel}/topk_entered",
                        points=(local_xyz + offsets[panel])[:1],
                        colors=np.asarray([[30, 255, 120]], dtype=np.uint8),
                        point_size=0.05,
                        visible=False,
                    )
                object_text = ""
            else:
                slot = OBJECT_ORDER.index(metric_name)
                changes = topk_change_masks(base, current, instance_targets[slot], object_masks[slot])
                left_count = int(changes["left"].sum())
                entered_count = int(changes["entered"].sum())
                for panel in (2, 3, 4):
                    panel_points = local_xyz + offsets[panel]
                    for key, color in (("left", (255, 30, 60)), ("entered", (30, 255, 120))):
                        selected = panel_points[changes[key]]
                        visible = bool(selected.shape[0])
                        if not visible:
                            selected = panel_points[:1]
                        server.scene.add_point_cloud(
                            f"/panel_{panel}/topk_{key}",
                            points=selected,
                            colors=np.repeat(np.asarray(color, dtype=np.uint8)[None], selected.shape[0], axis=0),
                            point_size=min(float(point_size.value) * 2.5, 0.12),
                            point_shape="circle",
                            precision="float32",
                            visible=visible,
                        )
                before = base_row["per_prompt_metrics"][prompt_index]["instances"][metric_name]
                after = candidate_row["per_prompt_metrics"][prompt_index]["instances"][metric_name]
                object_text = (
                    "**{}:** Top-k `{:.6f} → {:.6f}` · soft recall `{:.6f} → {:.6f}` · "
                    "active MAE `{:.6f} → {:.6f}`  \n"
                ).format(
                    OBJECT_LABELS[metric_name],
                    float(before["topk_overlap"]),
                    float(after["topk_overlap"]),
                    float(before["soft_recall"]),
                    float(after["soft_recall"]),
                    float(before["active_support_mae"]),
                    float(after["active_support_mae"]),
                )
                topk_text = "Top-k 교체점: **빨강=Base에서 빠짐 {}개**, **초록=Candidate에 들어옴 {}개**".format(
                    left_count, entered_count
                )
            status.content = (
                "**현재 표시:** `{}` · `{}` (t={}) · `{}` · `{}`  \n"
                "후보 eligible: `{}` · failed checks: `{}`  \n"
                "{}"
                "Panel prompt invariance: `{:.9f} → {:.9f}` · negative mean: `{:.9f} → {:.9f}`  \n"
                "Candidate−Base 실제 범위: `{:+.9f} … {:+.9f}` · 차이 색 범위: `±{:.9f}`  \n"
                "{}  \n"
                "**주의:** 이 화면은 one-step 진단 map이며 최종 500-step reverse-diffusion affordance map이 아닙니다."
            ).format(
                candidate_name,
                str(audit.value),
                int(report["audit_panels"][panel_index]["timestep"]),
                str(prompt.value),
                OBJECT_LABELS[focus_name],
                bool(row["eligible"]),
                ", ".join(str(value) for value in row["failed_checks"]),
                object_text,
                float(base_row["prompt_invariance"]),
                float(candidate_row["prompt_invariance"]),
                float(base_row["negative_mean"]),
                float(candidate_row["negative_mean"]),
                float(delta.min()),
                float(delta.max()),
                difference_limit,
                topk_text,
            )

    for control in (
        candidate,
        audit,
        prompt,
        focus,
        channel,
        render_mode,
        rgb_blend,
        heatmap_opacity,
        point_size,
    ):
        control.on_update(lambda _: render())
    render()
    print("[VIEWER_PASS] Teacher-v9.6 saved one-step maps loaded read-only")
    print("[PASS] Base/candidate/GT point order and response-map hash verified")
    print("[PASS] room_0101 only; no model, checkpoint, diffusion or training loaded")
    print("[OK] URL: http://{}:{}".format(args.host, args.port))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
