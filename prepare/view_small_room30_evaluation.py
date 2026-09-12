#!/usr/bin/env python3
"""Read saved maps only: no Torch, inference, GT interpolation, or training."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import socket
import threading
import time
import numpy as np
from small_room30_evaluation_common import validate_report, load_case

CHANNELS = ('any_joint', 'pelvis', 'left_foot', 'right_foot', 'neck', 'left_wrist', 'right_wrist')
TITLES = ('Input RGB', 'Contact GT', 'Frozen Base', 'Trained LoRA', 'LoRA - Base (red:+ blue:-)', '|LoRA - GT|')


def scalar(value, channel):
    return value.max(axis=1) if channel == 'any_joint' else value[:, CHANNELS.index(channel)-1]


def heat(values):
    # Fixed [0,1] scale shared by GT, Base, LoRA and error; never per-map rescaled.
    anchors = np.array([[.03, .03, .13], [.05, .22, .75], [0., .8, .85],
                        [.25, .9, .15], [1., .85, .05], [.85, .03, .02]])
    v = np.clip(values, 0., 1.)
    return np.round(np.stack([np.interp(v, np.linspace(0, 1, len(anchors)), anchors[:, i]) for i in range(3)], axis=1)*255).astype(np.uint8)


def panels(arrays, channel, blend=0.):
    rgb = arrays['points'][:, 3:].clip(0, 255).astype(np.uint8)
    gt = scalar(arrays['gt'], channel)
    base = scalar(arrays['base_raw'], channel)
    adapted = scalar(arrays['adapted_raw'], channel)
    # For any_joint, differences mean differences BETWEEN max-channel maps.
    diff = np.clip(adapted-base, -1, 1)
    signed = np.full((len(diff), 3), 235., dtype=np.float32)
    strength = np.abs(diff)[:, None]
    endpoint = np.where((diff >= 0)[:, None], np.array([230, 30, 20]), np.array([20, 80, 235]))
    signed = np.rint(signed*(1-strength) + endpoint*strength).astype(np.uint8)
    colors = [rgb, heat(gt), heat(base), heat(adapted), signed, heat(np.abs(adapted-gt))]
    for i in (1, 2, 3):
        colors[i] = np.rint((1-blend)*colors[i] + blend*rgb).astype(np.uint8)
    return colors


def save_png(arrays, channel, path, caption):
    from PIL import Image, ImageDraw, ImageFont
    if path.exists(): raise ValueError('PNG already exists; choose a new path')
    image = Image.new('RGB', (1500, 1090), 'white')
    draw = ImageDraw.Draw(image)
    try: font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError: font = ImageFont.load_default()
    draw.text((15, 10), caption, font=font, fill='black')
    draw.text((15, 34), 'Fixed map scale 0..1; difference -1..1. Raw metrics are NOT clipped. Development data only.', font=font, fill='black')
    xyz = arrays['points'][:, :3]
    xy = xyz[:, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = max(float((hi-lo).max()), 1e-6)
    positions = (xy - (lo+hi)/2) / span * 435
    order = np.argsort(xyz[:, 2])
    for i, colors in enumerate(panels(arrays, channel)):
        ox, oy = (i % 3)*500, 70+(i // 3)*500
        draw.text((ox+15, oy+10), TITLES[i], font=font, fill='black')
        for j in order:
            x, y = ox+250+positions[j, 0], oy+265-positions[j, 1]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=tuple(colors[j]))
    image.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8092)
    p.add_argument('--case', type=int, default=0)
    p.add_argument('--png-only', action='store_true')
    p.add_argument('--png-out', type=Path)
    args = p.parse_args()
    print('[VIEW 1/3] checking saved map hashes and recomputing metrics (no model load)', flush=True)
    report, manifest = validate_report(args.summary)
    rows = report['rows']
    if not 0 <= args.case < len(rows): raise ValueError('Invalid case index')
    root = args.summary.resolve().parent / 'cases'
    if args.png_out:
        row = rows[args.case]
        save_png(load_case(root / row['case']['filename']), 'any_joint', args.png_out,
                 '%s | %s | %s' % (manifest['scope'], row['case']['sample_id'], row['case']['generation']))
        print('[PNG] ' + str(args.png_out), flush=True)
    if args.png_only:
        if not args.png_out: raise ValueError('--png-only requires --png-out')
        return
    with socket.socket() as probe:
        probe.bind((args.host, args.port))  # Fail clearly rather than silently changing ports.
    import viser
    faulthandler.enable()

    def startup_timeout():
        print('[STOP] Viser constructor did not finish within 45s. Use --png-only; no model/training is running.', flush=True)
        faulthandler.dump_traceback()
        os._exit(2)  # Only this viewer process; never kill any external service.

    timer = threading.Timer(45., startup_timeout); timer.daemon = True; timer.start()
    print('[VIEW 2/3] starting Viser at %s:%d (45s startup timeout)' % (args.host, args.port), flush=True)
    try: server = viser.ViserServer(host=args.host, port=args.port)
    finally: timer.cancel()
    if hasattr(server, 'get_port') and server.get_port() != args.port:
        server.stop(); raise RuntimeError('Viser chose a different port; stop and choose a free port')
    server.scene.set_up_direction('+z')
    server.gui.add_markdown('## Saved Teacher rollout comparison\n%s\n\n'
        'Checkpoint step: **%s**. **Not Teacher approval. No training or inference.**\n\n'
        'GT/Base/LoRA colors share [0,1]; red/blue difference shares [-1,1]. '
        'Original 8192 points, no smoothing. Raw metrics retain out-of-range values.' % (manifest['scope'], manifest['checkpoint_step']))
    scene_ids = sorted({r['case']['scene_id'] for r in rows})
    scene = server.gui.add_dropdown('Scene', options=tuple(scene_ids), initial_value=rows[args.case]['case']['scene_id'])

    def choices(s):
        return {r['case']['sample_id'] + ' / g' + str(r['case']['generation']): r
                for r in rows if r['case']['scene_id'] == s}

    options = choices(scene.value)
    initial_case = rows[args.case]['case']
    selected = server.gui.add_dropdown('Prompt / generation', options=tuple(options),
        initial_value=initial_case['sample_id'] + ' / g' + str(initial_case['generation']))
    channel = server.gui.add_dropdown('Body channel', options=CHANNELS, initial_value='any_joint')
    point_size = server.gui.add_slider('Point size (m)', min=.005, max=.08, step=.005, initial_value=.025)
    blend = server.gui.add_slider('RGB blend', min=0., max=.5, step=.05, initial_value=0.)
    details = server.gui.add_markdown('')
    lock = threading.RLock()

    def render(_=None):
        with lock:
            row = choices(scene.value).get(selected.value)
            if row is None: return
            arrays = load_case(root / row['case']['filename'])
            xyz = arrays['points'][:, :3].copy()
            xyz[:, :2] -= (xyz[:, :2].min(axis=0) + xyz[:, :2].max(axis=0))/2
            extent = max(float(np.ptp(xyz[:, :2], axis=0).max()), 1.) + 1.2
            for i, colors in enumerate(panels(arrays, channel.value, float(blend.value))):
                offset = np.array([(i % 3 - 1)*extent, (0.5-i//3)*extent, 0], dtype=np.float32)
                server.scene.add_point_cloud('/panel%d/points' % i, points=xyz+offset, colors=colors,
                                             point_size=float(point_size.value))
                server.scene.add_label('/panel%d/title' % i, TITLES[i],
                    position=offset + np.array([0, extent*.43, max(float(xyz[:, 2].max()), 0)+.15]))
            scores = row['metrics']
            details.content = ('**%s**\n\nBase MAE: %.6f → LoRA: %.6f; zero: %.6f\n\n'
                               'LoRA per-motion active support:\n\n%s' % (
                row['case']['sample_id'], scores['base_raw']['mae_raw'], scores['adapted_raw']['mae_raw'],
                scores['zero_baseline']['mae_raw'], '\n\n'.join('%s: MAE %.4f, hit@0.3 %.3f' %
                (r['id'], r['active_mae_raw'], r['active_hit_rate_at_0_3']) for r in scores['adapted_raw']['instances'])))

    def change_scene(_):
        with lock:
            opts = choices(scene.value)
            selected.options = tuple(opts)
            selected.value = list(opts)[0]
            render()

    scene.on_update(change_scene)
    for control in (selected, channel, point_size, blend): control.on_update(render)

    @server.on_client_connect
    def connected(client):
        client.camera.position = (0., -14., 23.)
        client.camera.look_at = (0., 0., 0.)
        client.camera.up_direction = (0., 0., 1.)

    render()
    print('[VIEW 3/3] READY: http://127.0.0.1:%d (forward remote port %d in VS Code)' % (args.port, args.port), flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: server.stop()


if __name__ == '__main__': main()
