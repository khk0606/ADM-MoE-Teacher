#!/usr/bin/env python3
"""Read-only student viewer with verified observed history and explicit prompts."""
import argparse
import faulthandler
import os
import socket
import threading
import time
import inspect
from pathlib import Path
import numpy as np
from small_room30_anywhere_review import ReviewSource, overlay_geometry, details_markdown, PURPOSE_KO, TEXTS
from view_small_room30_evaluation import heat, scalar, CHANNELS

LAYOUTS = ('핵심 3개: RGB / Teacher / Final', 'Final 크게 보기', '전체 6개 지도', 'RGB + History 입력 확대', '이전 결과 / Teacher / 새 결과')
START = (220, 20, 230)
CURRENT = (0, 210, 240)
VELOCITY = (255, 125, 0)


class StudentViewer:
    def __init__(self, server, source, banner=''):
        self.server = server; self.source = source
        self.lock = threading.RLock(); self.busy = False; self.handles = []
        server.scene.set_up_direction('+z')
        server.gui.configure_theme(control_width='large', control_layout='fixed')
        server.gui.add_markdown('## Anywhere + Purpose Student\n\n' + banner +
            '\n\n**표시만 변경합니다. 재학습·추론·원본 수정 없음.**\n\n'
            '색상값: 파랑 0 → 초록 → 노랑 → 빨강 1. '
            '후보 점수 S = 0.7R + 0.3H → q = softmax(S/0.1). 공간 지지영역 × q = w; 최종 A_w = A × w. **q는 정답 확률이 아닙니다.**')
        self.scene = server.gui.add_dropdown('1. Scene / 방', options=source.scenes)
        self.prompt = server.gui.add_dropdown('2. 목적 prompt', options=source.options(self.scene.value))
        self.motion = server.gui.add_dropdown('3. History 출처', options=source.options(self.scene.value, self.prompt.value))
        self.generation = server.gui.add_dropdown('4. Generation', options=source.options(self.scene.value, self.prompt.value, self.motion.value))
        self.info = server.gui.add_markdown('')
        self.scores = server.gui.add_markdown('')
        self.layout = server.gui.add_dropdown('화면 구성', options=LAYOUTS)
        self.channel = server.gui.add_dropdown('Body channel', options=CHANNELS)
        self.size = server.gui.add_slider('점 크기 (m)', min=.005, max=.08, step=.005, initial_value=.025)
        self.show = server.gui.add_checkbox('History 표시', initial_value=True)
        self.end_heading = server.gui.add_checkbox('현재 몸 방향도 표시', initial_value=False)
        self.velocity = server.gui.add_checkbox('현재 이동 속도 화살표', initial_value=True)
        self.heading_length = server.gui.add_slider('몸 방향 화살표 길이 (m)', min=.2, max=1.5, step=.1, initial_value=.8)
        self.seconds = server.gui.add_slider('속도 화살표: v × 초', min=.1, max=1., step=.1, initial_value=.5)
        self.reset = server.gui.add_button('카메라 정렬 / Reset view')
        self.legend = server.gui.add_markdown(
            '### History 범례\n\n'
            '- **자홍색 START:** 0프레임 pelvis 위치 + 시작 몸 방향 화살표\n'
            '- **청록색 NOW:** 7프레임 pelvis 위치 (0.35초)\n'
            '- **흰/검정 궤적:** 관측된 8프레임의 pelvis 경로\n'
            '- **주황색 화살표:** 현재 XY 속도 × 지정 시간. 미래 궤적 예측이 아님\n\n'
            '몸 방향 화살표 길이는 표시용이며 속력이 아닙니다. '
            '궤적은 실제 크기를 유지합니다. 짧은 관측에서는 START/NOW가 겹칠 수 있습니다. '
            '시작 위치는 물체 중심이 아니라 **사람의 pelvis 높이**에 표시됩니다.')
        self.notice = server.gui.add_markdown('')
        with server.gui.add_folder('History 좌표·방향 수치 / 이 방의 모든 prompt', expand_by_default=False):
            self.numbers = server.gui.add_markdown('')
        self.scene.on_update(lambda _: self.refresh_options('scene'))
        self.prompt.on_update(lambda _: self.refresh_options('prompt'))
        self.motion.on_update(lambda _: self.refresh_options('motion'))
        self.generation.on_update(self.render)
        for handle in (self.channel, self.size, self.show, self.end_heading, self.velocity,
                       self.heading_length, self.seconds):
            handle.on_update(self.render)
        self.layout.on_update(lambda _: self.change_layout())
        self.reset.on_click(lambda _: self.reset_cameras())
        server.on_client_connect(self.reset_camera)
        self.render()

    def refresh_options(self, changed):
        with self.lock:
            if self.busy: return
            self.busy = True
            try:
                if changed == 'scene':
                    self.set_options(self.prompt, self.source.options(self.scene.value))
                if changed in ('scene', 'prompt'):
                    self.set_options(self.motion, self.source.options(self.scene.value, self.prompt.value))
                self.set_options(self.generation, self.source.options(self.scene.value, self.prompt.value, self.motion.value))
            finally:
                self.busy = False
            self.render()

    @staticmethod
    def set_options(control, options):
        old = control.value
        control.options = tuple(options)
        control.value = old if old in options else options[0]

    def change_layout(self):
        self.render(); self.reset_cameras()

    def reset_camera(self, client):
        mode = self.layout.value
        center = (7., 0., 0.) if mode in (LAYOUTS[0], LAYOUTS[4]) else ((7., -3.5, 0.) if mode == LAYOUTS[2] else (0., 0., 0.))
        distance = 20. if mode in (LAYOUTS[0], LAYOUTS[2], LAYOUTS[4]) else 6.
        client.camera.position = (center[0], center[1]-distance*.5, distance)
        client.camera.look_at = center
        client.camera.up_direction = (0., 1., 0.)

    def reset_cameras(self):
        for client in self.server.get_clients().values(): self.reset_camera(client)

    def keep(self, handle):
        self.handles.append(handle)

    def lines(self, name, points, color, thickness=.025):
        if not len(points): return
        method = self.server.scene.add_line_segments
        # Viser versions differ in line thickness parameter naming.
        params = inspect.signature(method).parameters
        kwargs = {'thickness': thickness} if 'thickness' in params else {'line_width': 3.}
        self.keep(method(name, points=points.astype(np.float32), colors=color, **kwargs))

    def history_overlay(self, index, info, center, offset):
        scene = self.server.scene
        g = overlay_geometry(info['prefix'], info['history'], center, offset,
                             self.heading_length.value, self.seconds.value)
        root = '/h%d' % index
        # Black outline and white core improve contrast on both high/low maps.
        self.lines(root+'/path_outline', g['trajectory'], (15,15,15), .055)
        self.lines(root+'/path', g['trajectory'], (255,255,255), .022)
        self.keep(scene.add_point_cloud(root+'/observed_frames', points=g['path'],
                                       colors=(255,255,255), point_size=.065))
        for label, point, color, displacement, anchor in (
                ('START t=0.00s', g['path'][0], START, [-.2,-.1,.18], 'center-right'),
                ('NOW t=0.35s', g['path'][-1], CURRENT, [.2,.1,.32], 'center-left')):
            self.keep(scene.add_icosphere(root+'/'+label.split()[0], radius=.075, color=color, position=point))
            kwargs = {}
            if 'anchor' in inspect.signature(scene.add_label).parameters:
                kwargs['anchor'] = anchor
            elif anchor == 'center-right':
                displacement = [-2.,-.1,.18]
            self.keep(scene.add_label(root+'/'+label.split()[0]+'_text', label, position=point+displacement, **kwargs))
        self.lines(root+'/start_body_heading', g['start_heading'], START)
        if self.end_heading.value: self.lines(root+'/current_body_heading', g['end_heading'], CURRENT)
        if self.velocity.value: self.lines(root+'/velocity', g['velocity'], VELOCITY)

    def render(self, _=None):
        with self.lock:
            if self.busy: return
            selection = (self.scene.value, self.prompt.value, self.motion.value, int(self.generation.value))
            if selection not in self.source.lookup: return
            arrays, info, record = self.source.load(selection)
            self.info.content = ('**Student 목적:** %s\n\n> %s\n\n'
                '**Teacher action:** %s\n\n'
                '**History:** `%s` · 관측 0.00 → 0.35 s') % (
                    PURPOSE_KO[record['prompt_id']], TEXTS[record['prompt_id']],
                    info['teacher_text'], record['history_motion_id'])
            d = self.source.diagnostics[selection]
            expected = {int(np.argmax(x['selection_target'])) for x in self.source.diagnostics.values()
                if x['record']['scene_id']==record['scene_id'] and x['record']['prompt_id']==record['prompt_id']}
            self.scores.content = ('**예측 후보 점수** (가구 정답 ID가 아닌 위치 기반 slot)\n\n'
                '| Slot | R | H | 0.7R+0.3H | 경쟁 q |\n|---|---:|---:|---:|---:|\n' +
                '\n'.join('| %d | %.3f | %.3f | %.3f | %.3f |' % (j+1,
                    arrays['relation_score'][j], arrays['history_score'][j],
                    arrays['score'][j], arrays['selection'][j]) for j in range(2)) +
                '\n\n**자동 진단:** ' + (', '.join(d['failures']) if d['failures'] else '검사 통과 — 시각 승인 필요') +
                '\n\n후보 위치 오차(m): ' + ', '.join('%.3f'%x for x in d['slot_error_m']) +
                '\n\n바닥 출력/목표 비율: ' + str(d['floor_predicted_to_target_ratio']))
            self.scores.content += ('\n\n**이 prompt의 원본 history 쌍:** ' +
                ('목표 후보 전환이 예상됨' if len(expected)>1 else '같은 후보를 선호하는 목표 — 무조건 전환을 요구하지 않음'))
            self.numbers.content = details_markdown(record, info, self.source.options(self.scene.value))
            speed = float(np.linalg.norm(info['history'][-1,2:4]))
            self.notice.content = ('현재 속력 **%.4f m/s**. %s\n\n'
                '같은 방·목적에서 History만 바꾸면 W_R은 같고 W_H가 어떻게 달라지는지 비교하세요. '
                'Learning target은 원본 Contact GT가 아닙니다. 같은 방 g2 평가이며 새로운 방 일반화 평가는 아닙니다.') % (
                    speed, '속도가 0이어서 주황색 화살표를 그리지 않습니다.' if speed < 1e-6 else
                    '주황색 화살표 길이 = %.3f m (표시 배율 포함).' % (speed*self.seconds.value))
            xyz = arrays['points'][:,:3]; center = xyz.mean(0)
            channel = self.channel.value
            entries = [('Teacher A (1578)', heat(scalar(arrays['teacher_a'], channel))),
                ('Relation score on support', heat(arrays['relation_map'])), ('History score on support', heat(arrays['history_map'])),
                ('Spatial weight after competition', heat(arrays['w'])), ('FINAL A_w', heat(scalar(arrays['a_w'], channel))),
                ('Learning target (NOT raw GT)', heat(scalar(arrays['training_target_aw'], channel)))]
            rgb = ('RGB + observed history', np.rint(arrays['points'][:,3:].clip(0,1)*255).astype(np.uint8))
            mode = self.layout.value
            if mode == LAYOUTS[0]: entries = [rgb, entries[0], entries[4]]
            elif mode == LAYOUTS[1]: entries = [entries[4]]
            elif mode == LAYOUTS[3]: entries = [rgb]
            elif mode == LAYOUTS[4]: entries = [(('Teacher A — 이전 anywhere Student 없음' if bool(arrays['baseline_is_teacher']) else 'Previous competition A_w'), heat(scalar(arrays['baseline_aw'], channel))), entries[0], entries[4]]
            for handle in self.handles: handle.remove()
            self.handles = []
            for i, (title, colors) in enumerate(entries):
                offset = np.array([(i%3)*7., -(i//3)*7., 0.])
                self.keep(self.server.scene.add_point_cloud('/map%d'%i, points=xyz-center+offset,
                                                           colors=colors, point_size=self.size.value))
                self.keep(self.server.scene.add_label('/title%d'%i, title, position=offset+[0,3.4,1.]))
                if self.show.value: self.history_overlay(i, info, center, offset)
                for j, xy in enumerate(arrays['slot_xy']):
                    pos = np.array([xy[0], xy[1], np.quantile(xyz[:,2], .1)+.8])-center+offset
                    self.keep(self.server.scene.add_label('/slot%d_%d'%(i,j), 'Slot %d · q=%.3f'%(j+1,arrays['selection'][j]), position=pos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('outputs/small_room30_student_anywhere01/summary.json'))
    parser.add_argument('--dataset', type=Path, default=Path('data/small_room30_adm_lora_v1'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8100)
    args = parser.parse_args()
    print('[VERIFY] Student maps + bound dataset + exact prompt/history source. No training.', flush=True)
    source = ReviewSource(args.summary, args.dataset)
    try:
        with socket.socket() as s: s.bind((args.host, args.port))
    except OSError as exc:
        raise SystemExit('Port %d is occupied. Choose --port or PORT=8099; do not kill unrelated servers. %s' % (args.port, exc))
    import viser
    faulthandler.enable()
    def timeout():
        print('[STOP] Viser startup exceeded 45s. No training is running.', flush=True)
        faulthandler.dump_traceback(); os._exit(2)
    timer = threading.Timer(45, timeout); timer.daemon = True; timer.start()
    try: server = viser.ViserServer(host=args.host, port=args.port)
    finally: timer.cancel()
    if hasattr(server, 'get_port') and server.get_port() != args.port:
        server.stop(); raise RuntimeError('Requested port unavailable')
    StudentViewer(server, source)
    print('[READY] http://%s:%d — forward this port; Ctrl-C stops viewer only' % (args.host,args.port), flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__': main()

