"""Read-only competition maps with exact formula and observed-history checks."""
import json
from pathlib import Path
import numpy as np
from small_room30_adm_dataset import SmallRoom30ADM, inside, sha
from relation_aware_moe_v2_contract import observed_history_prefix_numpy
from small_room30_student_data import TEXTS
from small_room30_competition_data import POLICY

PURPOSE_KO = {
    'sit_watch_v1': 'TV를 보기 위해 앉기',
    'sit_write_v1': '책상 또는 화이트보드 근처에 앉아 쓰기',
    'sit_write_desk_v1': '책상 근처에 앉아 쓰기',
    'sit_write_whiteboard_v1': '화이트보드 근처에 앉아 쓰기',
}


def key(record):
    return tuple(record[k] for k in ('scene_id', 'prompt_id', 'history_motion_id', 'generation'))


def arrow_segments(origin, vector):
    """XY arrow with actual supplied length; empty for a stationary vector."""
    origin = np.asarray(origin, np.float32)
    vector = np.asarray(vector, np.float32)
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return np.empty((0, 2, 3), np.float32)
    unit = vector / length
    side = np.array([-unit[1], unit[0], 0], np.float32)
    tip = origin + vector
    head = min(.18, length * .28)
    return np.asarray([[origin, tip], [tip, tip-head*unit+.5*head*side],
                       [tip, tip-head*unit-.5*head*side]], np.float32)


def overlay_geometry(prefix, history, scene_center, offset, heading_length=.8, velocity_seconds=.5):
    """Identical scene transform for points and pelvis; no re-centering the path."""
    path = np.asarray(prefix, np.float32)[:, 0, :] - scene_center + offset
    start_vector = np.r_[history[0, 4:6] * heading_length, 0]
    end_vector = np.r_[history[-1, 4:6] * heading_length, 0]
    velocity = np.r_[history[-1, 2:4] * velocity_seconds, 0]
    return dict(path=path, trajectory=np.stack((path[:-1], path[1:]), axis=1),
        start_heading=arrow_segments(path[0], start_vector),
        end_heading=arrow_segments(path[-1], end_vector),
        velocity=arrow_segments(path[-1], velocity))


class ReviewSource:
    def __init__(self, summary, dataset):
        self.summary = Path(summary).resolve(); self.root = self.summary.parent
        self.report = json.loads(self.summary.read_text())
        if self.report.get('status') != 'COMPETITION_COMPLETE_REVIEW_REQUIRED':
            raise ValueError('Need completed competition summary; old student maps are incompatible')
        if self.report.get('policy') != POLICY or self.report.get('approved') is not False:
            raise ValueError('Wrong competition policy/review status')
        manifest_path = self.root / 'manifest.json'
        if sha(manifest_path) != self.report['manifest_sha256']:
            raise ValueError('Student manifest changed')
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get('schema') != 'small_room30_competition_run_v1' or self.manifest.get('policy') != POLICY:
            raise ValueError('Wrong competition manifest')
        if self.manifest.get('prompt_texts') != TEXTS:
            raise ValueError('Prompt text mismatch')
        if sha(self.root/'comparison.json') != self.report['comparison_sha256']:
            raise ValueError('Diagnostics changed')
        diagnostics = json.loads((self.root/'comparison.json').read_text())
        self.diagnostics = {key(r['record']): r for r in diagnostics['rows']}
        self.data = SmallRoom30ADM(dataset)
        binding = self.manifest['binding']
        if (sha(self.data.root/'index.json') != binding['dataset_index_sha256'] or
                sha(self.data.root/'SHA256SUMS.txt') != binding['dataset_manifest_sha256']):
            raise ValueError('Not the dataset used by this student run')
        # Prompts were stored in source, not old map NPZ. Verify that exact source
        # before showing TEXTS as the text actually encoded during training.
        source = Path(__file__).parent
        for name in ('small_room30_student_data.py', 'relation_aware_moe_v2_contract.py', 'small_room30_competition_data.py', 'small_room30_competition_review.py'):
            if sha(source/name) != self.manifest['code_sha256'][name]:
                raise ValueError('Training prompt/history code changed: '+name)
        expected = [r for r in binding['records'] if r['split'] != 'train']
        self.rows = self.report['rows']
        if [r['record'] for r in self.rows] != expected or len({key(r) for r in expected}) != len(expected):
            raise ValueError('Evaluation records differ from student manifest')
        self.lookup = {key(r['record']): r for r in self.rows}
        self.scenes = sorted({r['record']['scene_id'] for r in self.rows})
        self.contact_root = inside(self.data.root, self.data.index['source_contact_index']).parent
        self.motions = {m['id']:m for m in json.loads((self.contact_root/'index.json').read_text())['motions']}
        self.cache = {}
        for row in self.rows:
            if sha(inside(self.root, row['file'])) != row['sha256']:
                raise ValueError('Saved map changed: '+row['file'])

    def options(self, scene, prompt=None, motion=None):
        records = [r['record'] for r in self.rows if r['record']['scene_id'] == scene]
        if prompt is None: return sorted({r['prompt_id'] for r in records})
        records = [r for r in records if r['prompt_id'] == prompt]
        if motion is None: return sorted({r['history_motion_id'] for r in records})
        return sorted({str(r['generation']) for r in records if r['history_motion_id'] == motion})

    def load(self, selection):
        row = self.lookup[selection]; record = row['record']
        path = inside(self.root, row['file'])
        if sha(path) != row['sha256']: raise ValueError('Saved map changed')
        with np.load(path, allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        if arrays['points'].shape != (8192,6): raise ValueError('Wrong map shape')
        identity = (record['scene_id'], record['history_motion_id'])
        if identity not in self.cache:
            motion = self.motions[record['history_motion_id']]
            if motion['scene_id'] != record['scene_id']: raise ValueError('History belongs to another room')
            sample = next(s for s in self.data.samples if s['id'] == record['sample_id'])
            if (sample['scene_id'] != record['scene_id'] or motion['id'] not in sample['source_motion_ids']):
                raise ValueError('History/sample mismatch')
            motion_path = inside(self.contact_root, motion['file'])
            if sha(motion_path) != motion['sha256']: raise ValueError('Recorded history source changed')
            with np.load(motion_path, allow_pickle=False) as z:
                poses = z['motion_smpl22_20fps']
                history = observed_history_prefix_numpy(poses, np.arange(len(poses))/20.)
                prefix = poses[:8].copy()
            points_path = inside(self.data.root, self.data.scenes[record['scene_id']]['points_file'])
            with np.load(points_path, allow_pickle=False) as z:
                expected_points = z['points'].copy(); expected_points[:,3:] /= 255.
            self.cache[identity] = dict(history=history, prefix=prefix, expected_points=expected_points,
                teacher_text=sample['text'], motion_sha256=motion['sha256'])
        info = self.cache[identity]
        if not np.array_equal(arrays['points'], info['expected_points']):
            raise ValueError('History overlay and saved scene have different point coordinates/order')
        if not np.array_equal(arrays['observed_history'], info['history']):
            raise ValueError('Saved and reconstructed observed history differ')
        validate_maps(arrays)
        return arrays, info, record


def details_markdown(record, info, scene_prompts):
    h = info['history']; start = info['prefix'][0,0]; end = info['prefix'][-1,0]
    heading = np.degrees(np.arctan2(h[:,5],h[:,4]))
    speed = np.linalg.norm(h[-1,2:4]); displacement = np.linalg.norm(end[:2]-start[:2])
    return ('### 현재 선택 / Current example\n\n'
        '**방:** `%s` · **generation:** g%s\n\n'
        '**Student 목적:** %s\n\n> %s\n\n'
        '**Teacher action prompt (q_a):**\n\n> %s\n\n'
        '**이 방에 저장된 목적:**\n\n%s\n\n'
        '**History 원본:** `%s`\n\n'
        '이 ID는 입력 궤적의 출처이며, 예측된 선택 가구가 아닙니다.\n\n'
        '**관측 구간:** 0–7 frame · 20 fps · 0.00–0.35 s · 미래 궤적 표시 안 함\n\n'
        '**시작 pelvis XYZ (m):** (%.3f, %.3f, %.3f)\n\n'
        '**현재 pelvis XYZ (m):** (%.3f, %.3f, %.3f)\n\n'
        '**시작 몸 방향 XY:** (%.3f, %.3f) / %.1f°\n\n'
        '**현재 몸 방향:** %.1f° · **현재 속력:** %.3f m/s\n\n'
        '**관측 XY 이동거리(시작→현재):** %.3f m\n\n'
        '각도는 원본 AMDM Z-up 좌표의 +X=0°, +Y=90°. '
        '몸 방향은 이동 방향과 다를 수 있습니다. '
        '시작 프레임 속도는 이전 관측이 없어 입력 규칙상 0으로 설정됩니다.'
        ) % (record['scene_id'], record['generation'], PURPOSE_KO[record['prompt_id']],
             TEXTS[record['prompt_id']], info['teacher_text'],
             '\n'.join('- %s — %s' % (PURPOSE_KO[p], TEXTS[p]) for p in scene_prompts),
             record['history_motion_id'], *start, *end, *h[0,4:6], heading[0], heading[-1], speed, displacement)

def validate_maps(a):
    n=8192
    shapes=dict(points=(n,6),teacher_a=(n,6),a_w=(n,6),baseline_aw=(n,6),
        training_target_aw=(n,6),observed_history=(8,6),support=(n,3),
        slot_xy=(2,2),context_xy=(3,2),selection=(2,),score=(2,),
        relation_score=(2,),history_score=(2,),w=(n,),
        relation_map=(n,),history_map=(n,),score_map=(n,))
    for key,shape in shapes.items():
        if key not in a or a[key].shape!=shape or not np.isfinite(a[key]).all():
            raise ValueError('Invalid saved array: '+key)
    for key in ('support','selection','relation_score','history_score','w'):
        if (a[key]<-1e-6).any() or (a[key]>1+1e-6).any():
            raise ValueError('Out of range: '+key)
    score=.7*a['relation_score']+.3*a['history_score']
    q=np.exp((score-score.max())/.1);q/=q.sum()
    w=(a['support'][:,:2]*q).sum(-1)
    expected=dict(score=score,selection=q,w=w,a_w=a['teacher_a']*w[:,None],
        relation_map=(a['support'][:,:2]*a['relation_score']).sum(-1),
        history_map=(a['support'][:,:2]*a['history_score']).sum(-1),
        score_map=(a['support'][:,:2]*score).sum(-1))
    if not np.allclose(a['support'].sum(-1),1,atol=2e-5):
        raise ValueError('Invalid support normalization')
    for key,value in expected.items():
        if not np.allclose(a[key],value,atol=2e-5,rtol=2e-5):
            raise ValueError('Competition formula mismatch: '+key)

