#!/usr/bin/env python3
"""Read-only six-panel viewer for saved Teacher-v9.8.12 rollout maps."""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from relational_teacher_v9812_affordance_viewer_common import (
    CHANNEL_ORDER,
    affordance_colors,
    build_xy_interpolation_plan,
    interpolate_xy_heatmap,
    rgba_heatmap,
    scalar_channel,
    signed_difference_colors,
)


SCHEMA = "relational_teacher_v9812_two_scene_multiupdate_calibration_v1"
SCENES = ("room_0101", "room_0102")
PROMPTS = ("watch", "write")
ROLES = ("bed", "normal_chair", "high_chair")
ROLE_LABELS = {
    "all_verified": "All verified Sit objects",
    "bed": "Bed",
    "normal_chair": "Normal Chair",
    "high_chair": "High Chair",
}
POINT_COUNT = 8192
UPDATE_COUNT = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Teacher-v9.8.12 actual-rollout affordance audit."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
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
        or report.get("shortlisted_steps") != []
        or report.get("failed_checks")
        != ["at_least_one_all_three_state_is_shortlisted"]
        or report.get("update_count") != UPDATE_COUNT
        or report.get("source_scene") != SCENES[0]
        or report.get("audit_scene") != SCENES[1]
        or report.get("serialized_model_state") is not False
        or report.get("development_arrays_read") is not False
        or report.get("paper_test_access") is not False
    ):
        raise ValueError("viewer rejected the Teacher-v9.8.12 report authority")
    rows = report.get("monitor_rows")
    if (
        not isinstance(rows, list)
        or len(rows) != UPDATE_COUNT
        or [row.get("step") for row in rows] != list(range(1, UPDATE_COUNT + 1))
        or any(row.get("eligible") is not False for row in rows)
    ):
        raise ValueError("Teacher-v9.8.12 monitor inventory changed")
    paths = report.get("paths")
    hashes = report.get("path_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Teacher-v9.8.12 path binding is absent")
    maps_file = Path(str(paths.get("calibration_maps", ""))).expanduser().resolve()
    if (
        not maps_file.is_file()
        or sha256_file(maps_file) != hashes.get("calibration_maps")
        or sha256_file(maps_file) != report.get("calibration_maps_sha256")
    ):
        raise ValueError("Teacher-v9.8.12 map hash changed")
    return report, maps_file


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    required = {
        "scene_ids",
        "prompt_ids",
        "base",
        "start",
        "candidates",
        "source_xyz",
        "source_points",
        "source_instance_names",
        "source_verified_object_mask",
        "source_instance_targets",
        "source_all_sittable_gt",
        "audit_xyz",
        "audit_points",
        "audit_instance_names",
        "audit_verified_object_mask",
        "audit_instance_targets",
        "audit_all_sittable_gt",
    }
    if not required.issubset(arrays):
        raise ValueError("Teacher-v9.8.12 viewer arrays are incomplete")
    map_shape = (2, 3, 2, POINT_COUNT, 6)
    if (
        arrays["scene_ids"].tolist() != list(SCENES)
        or arrays["prompt_ids"].tolist() != ["sit_watch_v1", "sit_write_v1"]
        or arrays["base"].shape != map_shape
        or arrays["start"].shape != map_shape
        or arrays["candidates"].shape != (UPDATE_COUNT,) + map_shape
    ):
        raise ValueError("Teacher-v9.8.12 map shape/order changed")
    for prefix in ("source", "audit"):
        if (
            arrays[prefix + "_xyz"].shape != (POINT_COUNT, 3)
            or arrays[prefix + "_points"].shape != (POINT_COUNT, 6)
            or arrays[prefix + "_verified_object_mask"].shape != (3, POINT_COUNT)
            or arrays[prefix + "_instance_targets"].shape != (3, POINT_COUNT, 6)
            or arrays[prefix + "_all_sittable_gt"].shape != (POINT_COUNT, 6)
            or arrays[prefix + "_instance_names"].shape != (3,)
        ):
            raise ValueError(prefix + " scene bundle shape changed")
        if not np.array_equal(
            arrays[prefix + "_xyz"], arrays[prefix + "_points"][:, :3]
        ):
            raise ValueError(prefix + " XYZ/point order differs")
    for name in (
        "base",
        "start",
        "candidates",
        "source_xyz",
        "audit_xyz",
        "source_instance_targets",
        "audit_instance_targets",
    ):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(name + " contains NaN/Inf")
    for name in ("base", "start", "candidates"):
        if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
            raise ValueError(name + " is outside [0,1]")


def _blend(map_colors: np.ndarray, rgb: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.6))
    return np.rint((1.0 - amount) * map_colors + amount * rgb).clip(0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    report_file = args.summary.expanduser().resolve()
    report, maps_file = _validate_report(report_file)
    arrays = load_npz(maps_file)
    _validate_arrays(arrays)

    import viser

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label="Teacher-v9.8.12 affordance audit",
    )
    server.gui.add_markdown(
        "## Teacher-v9.8.12 actual-rollout map audit\n"
        "**저장된 두 train 장면의 실제 K=3 결과를 읽기만 합니다. 재학습·재추론 없음.**  \n"
        "위: `Input RGB → All-sittable GT → radius-0.006 Start`  \n"
        "아래: `선택 Update → Update−Start → |Update−GT|`  \n"
        "연속 면은 표시 전용 XY 보간이며 수치 검사는 원본 8192점을 사용했습니다.  \n"
        "⚠️ 모든 update의 all-three가 0/3이라 이 결과는 **FAIL**이며 checkpoint가 없습니다."
    )
    scene_control = server.gui.add_dropdown(
        "Scene", options=SCENES, initial_value=SCENES[0]
    )
    update_control = server.gui.add_dropdown(
        "State", options=("start",) + tuple("update_{}".format(i) for i in range(1, 7)), initial_value="update_6"
    )
    generation_control = server.gui.add_dropdown(
        "Generation", options=("generation_0", "generation_1", "generation_2"), initial_value="generation_0"
    )
    prompt_control = server.gui.add_dropdown(
        "Prompt", options=PROMPTS, initial_value="watch"
    )
    focus_control = server.gui.add_dropdown(
        "Object focus", options=("all_verified",) + ROLES, initial_value="all_verified"
    )
    channel_control = server.gui.add_dropdown(
        "Affordance channel", options=("any_joint",) + CHANNEL_ORDER, initial_value="any_joint"
    )
    render_control = server.gui.add_dropdown(
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
    plans: Dict[str, Mapping[str, np.ndarray]] = {}

    def render() -> None:
        with lock:
            scene = str(scene_control.value)
            scene_index = SCENES.index(scene)
            prefix = "source" if scene == SCENES[0] else "audit"
            xyz = np.asarray(arrays[prefix + "_xyz"], dtype=np.float32)
            points = np.asarray(arrays[prefix + "_points"], dtype=np.float32)
            rgb = rgb_u8(points)
            center_xy = 0.5 * (xyz[:, :2].min(axis=0) + xyz[:, :2].max(axis=0))
            local_xyz = xyz.copy()
            local_xyz[:, :2] -= center_xy
            if scene not in plans:
                plans[scene] = build_xy_interpolation_plan(
                    local_xyz,
                    resolution=args.heatmap_resolution,
                    neighbors=args.heatmap_neighbors,
                )
            plan = plans[scene]
            width = float(np.ptp(local_xyz[:, 0]))
            depth = float(np.ptp(local_xyz[:, 1]))
            spacing_x = max(width + 1.0, 4.0)
            spacing_y = max(depth + 1.0, 4.0)
            offsets = (
                np.asarray((-spacing_x, 0.5 * spacing_y, 0.0), np.float32),
                np.asarray((0.0, 0.5 * spacing_y, 0.0), np.float32),
                np.asarray((spacing_x, 0.5 * spacing_y, 0.0), np.float32),
                np.asarray((-spacing_x, -0.5 * spacing_y, 0.0), np.float32),
                np.asarray((0.0, -0.5 * spacing_y, 0.0), np.float32),
                np.asarray((spacing_x, -0.5 * spacing_y, 0.0), np.float32),
            )
            generation = int(str(generation_control.value).split("_")[-1])
            prompt_index = PROMPTS.index(str(prompt_control.value))
            focus = str(focus_control.value)
            masks = np.asarray(arrays[prefix + "_verified_object_mask"], bool)
            targets = np.asarray(arrays[prefix + "_instance_targets"], np.float32)
            if focus == "all_verified":
                focus_mask = masks.any(axis=0)
                gt = np.asarray(arrays[prefix + "_all_sittable_gt"], np.float32)
            else:
                role_index = ROLES.index(focus)
                focus_mask = masks[role_index]
                gt = targets[role_index]
            start = np.asarray(
                arrays["start"][scene_index, generation, prompt_index], np.float32
            )
            state = str(update_control.value)
            if state == "start":
                update = start
                row = None
            else:
                step = int(state.split("_")[-1])
                update = np.asarray(
                    arrays["candidates"][step - 1, scene_index, generation, prompt_index],
                    np.float32,
                )
                row = report["monitor_rows"][step - 1]
            channel = str(channel_control.value)
            gt_scalar = scalar_channel(gt, channel)
            start_scalar = scalar_channel(start, channel)
            update_scalar = scalar_channel(update, channel)
            difference = update_scalar - start_scalar
            error = np.abs(update_scalar - gt_scalar)
            difference_limit = max(float(np.abs(difference).max()), 1e-8)
            maps = (None, gt_scalar, start_scalar, update_scalar, difference, error)
            names = (
                "Input 3D Scene (RGB)",
                ROLE_LABELS[focus] + " GT",
                "radius-0.006 Start",
                state.replace("_", " ").title(),
                "Update − Start",
                "|Update − GT|",
            )
            for panel, (name, values, offset) in enumerate(zip(names, maps, offsets)):
                panel_points = local_xyz + offset
                if panel == 0:
                    colors = rgb
                elif panel == 4:
                    colors = signed_difference_colors(values, difference_limit)
                else:
                    colors = _blend(affordance_colors(values), rgb, float(rgb_blend.value))
                point_visible = panel == 0 or str(render_control.value) != "continuous heatmap only"
                server.scene.add_point_cloud(
                    "/panel_{}/points".format(panel),
                    points=panel_points,
                    colors=colors,
                    point_size=float(point_size.value),
                    point_shape="circle",
                    precision="float32",
                    visible=point_visible,
                )
                if panel != 0:
                    raster = interpolate_xy_heatmap(values, plan)
                    heat_colors = (
                        signed_difference_colors(raster, difference_limit)
                        if panel == 4
                        else affordance_colors(raster)
                    )
                    xy_min = np.asarray(plan["xy_min"], np.float32)
                    xy_max = np.asarray(plan["xy_max"], np.float32)
                    position = offset + np.asarray(
                        (
                            0.5 * float(xy_min[0] + xy_max[0]),
                            0.5 * float(xy_min[1] + xy_max[1]),
                            float(np.asarray(plan["floor_z"]).item()) - 0.015,
                        ),
                        np.float32,
                    )
                    server.scene.add_image(
                        "/panel_{}/heatmap".format(panel),
                        image=rgba_heatmap(heat_colors, float(heatmap_opacity.value)),
                        render_width=float(xy_max[0] - xy_min[0]),
                        render_height=float(xy_max[1] - xy_min[1]),
                        position=position,
                        visible=str(render_control.value) != "points only",
                        cast_shadow=False,
                        receive_shadow=False,
                    )
                highlights = panel_points[focus_mask]
                server.scene.add_point_cloud(
                    "/panel_{}/focus".format(panel),
                    points=highlights,
                    colors=np.repeat(
                        np.asarray((35, 255, 90), np.uint8)[None],
                        highlights.shape[0],
                        axis=0,
                    ),
                    point_size=min(float(point_size.value) * 1.55, 0.09),
                    point_shape="circle",
                    precision="float32",
                    visible=bool(highlights.shape[0]),
                )
                server.scene.add_label(
                    "/panel_{}/label".format(panel),
                    text=name,
                    position=offset
                    + np.asarray(
                        (0.0, 0.0, float(local_xyz[:, 2].max()) + 0.55), np.float32
                    ),
                )

            if row is None:
                status.content = (
                    "**현재 표시:** `{}` · `start` · `generation {}` · `{}` · `{}`  \n"
                    "All-three: 시작 상태에서도 `watch 0/3`, `write 0/3`  \n"
                    "Start는 v9.8.11에서 통과한 radius-0.006 상태이지만 절대 세 객체 동시 검출은 실패했습니다."
                ).format(scene, generation, str(prompt_control.value), ROLE_LABELS[focus])
            else:
                pooled = row["scene_candidate_pooled"][scene]
                counts = row["all_three_counts"][scene]
                status.content = (
                    "**현재 표시:** `{}` · `update {}` · `generation {}` · `{}` · `{}`  \n"
                    "Update eligible: `{}` · strict scene failed: `{}`  \n"
                    "Pooled recall Bed/Normal/High: `{:.6f} / {:.6f} / {:.6f}`  \n"
                    "Pooled MAE Bed/Normal/High: `{:.6f} / {:.6f} / {:.6f}`  \n"
                    "**All-three:** watch `{}/3`, write `{}/3` — 절대 기준 실패"
                ).format(
                    scene,
                    row["step"],
                    generation,
                    str(prompt_control.value),
                    ROLE_LABELS[focus],
                    row["eligible"],
                    ", ".join(row["scene_failed_checks"][scene]) or "없음(상대 기준 통과)",
                    pooled["bed"]["soft_recall"],
                    pooled["normal_chair"]["soft_recall"],
                    pooled["high_chair"]["soft_recall"],
                    pooled["bed"]["active_support_mae"],
                    pooled["normal_chair"]["active_support_mae"],
                    pooled["high_chair"]["active_support_mae"],
                    counts["watch"],
                    counts["write"],
                )

    for control in (
        scene_control,
        update_control,
        generation_control,
        prompt_control,
        focus_control,
        channel_control,
        render_control,
        rgb_blend,
        heatmap_opacity,
        point_size,
    ):
        control.on_update(lambda _: render())
    render()
    print("[VIEWER_PASS] Teacher-v9.8.12 saved actual-rollout maps loaded read-only")
    print("[PASS] two scenes, Start, six updates, GT and map hash verified")
    print("[PASS] no model, diffusion, optimizer, checkpoint or room_0201 loaded")
    print("[OK] URL: http://{}:{}".format(args.host, args.port))
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
