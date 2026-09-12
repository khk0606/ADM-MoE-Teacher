"""1578 full-evaluation cache + small-room annotations, separated from forward.

The development pilot uses all 30 scenes. g0/g1 train, g2 is same-scene
noise-repeat evaluation, NOT a held-out scene/history generalization claim.
"""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from small_room30_adm_dataset import SmallRoom30ADM, inside, sha
from small_room30_evaluation_common import validate_report, panel, load_case
from relation_aware_moe_v2_contract import observed_history_prefix_numpy

TEACHER_SHA = '1b0b4da93e376e007c8e489bf390586d7108ecd46cc1cd523d9787b6a23e814a'
RUN_SHA = '422fab37049f978087f039aaa223446ac1fefbe23563e6f4817a6c67de8e0588'
TEXTS = {
    'sit_watch_v1': 'Sit on something to watch TV',
    'sit_write_v1': 'Sit on something near a desk or whiteboard to write',
    'sit_write_desk_v1': 'Sit on something near a desk to write',
    'sit_write_whiteboard_v1': 'Sit on something near a whiteboard to write',
}


def observed_scores(history, anchors):
    """Approach suitability from frame-7 state ONLY, not eventual contact/ID.

    One-metre distance scale + bounded heading/movement alignment; sigmoid
    relative to the mean candidate score. Does not alter relation ranking GT.
    """
    delta = np.asarray(anchors, np.float32) - history[-1, :2]
    distance = np.linalg.norm(delta, axis=1)
    direction = delta / np.maximum(distance[:, None], 1e-6)
    speed = np.linalg.norm(history[-1, 2:4])
    movement = history[-1, 2:4] / max(float(speed), 1e-6)
    scores = -distance + .5 * (direction @ history[-1, 4:6])
    scores += .5 * min(float(speed), 1.) * (direction @ movement)
    return (1 / (1 + np.exp(-(scores - scores.mean())))).astype(np.float32)


def spatial_targets(contacts, masks, selected, history_scores, purpose):
    affinity = contacts.max(-1)  # [C,N], all six channels; supervision only.
    denom = affinity.sum(0)
    relation = affinity[selected] / np.maximum(denom, 1e-6)
    historical = (affinity * history_scores[:, None]).sum(0) / np.maximum(denom, 1e-6)
    for c, mask in enumerate(masks):
        relation[mask] = float(c == selected)
        historical[mask] = history_scores[c]
    relation[purpose] = 0
    historical[purpose] = 0
    return relation.astype(np.float32), historical.astype(np.float32), affinity.max(0)


class StudentData:
    def __init__(self, dataset, teacher_summary):
        self.data = SmallRoom30ADM(dataset)
        self.summary_path = Path(teacher_summary).resolve()
        self.report, self.manifest = validate_report(self.summary_path)
        m = self.manifest
        if (m.get('mode') != 'full' or m.get('checkpoint_step') != 1578
                or m.get('checkpoint_sha256') != TEACHER_SHA
                or m.get('training_summary_sha256') != RUN_SHA
                or m['cases'] != panel(self.data, 'full')):
            raise ValueError('Need completed all-30-room, K=3 saved1578 evaluation; not v4/1560/canary')
        self.contact_root = inside(self.data.root, self.data.index['source_contact_index']).parent
        contact_index = json.loads((self.contact_root / 'index.json').read_text())
        self.motions = {r['id']: r for r in contact_index['motions']}
        self.case_rows = {r['case']['filename']: r for r in self.report['rows']}
        self.records = []
        for scene in sorted(self.data.scenes):
            sample = next(s for s in self.data.samples if s['scene_id'] == scene and s['prompt_id'] == 'sit_anywhere')
            relations = json.loads(inside(self.data.root, self.data.scenes[scene]['relation_labels']).read_text())
            for relation in relations['prompts']:
                if relation['prompt_id'] not in TEXTS:
                    raise ValueError('Unspecified purpose semantics: ' + relation['prompt_id'])
                for motion_id in sample['source_motion_ids']:
                    for g in range(3):
                        self.records.append(dict(scene_id=scene, sample_id=sample['id'],
                            prompt_id=relation['prompt_id'], history_motion_id=motion_id,
                            generation=g, teacher_file=sample['id'] + '__g%d.npz' % g,
                            split='train' if g < 2 else 'same_scene_noise_repeat'))
        if {r['scene_id'] for r in self.records} != set(self.data.scenes):
            raise ValueError('Some scenes lack usable relation/history examples')
        # Validate all scene/history/Teacher pairings before any optimizer exists.
        for r in self.records:
            self.example(r)
        self.binding = dict(teacher_step=1578, teacher_checkpoint_sha256=TEACHER_SHA,
            teacher_summary_sha256=sha(self.summary_path), dataset_index_sha256=sha(self.data.root / 'index.json'),
            dataset_manifest_sha256=sha(self.data.root / 'SHA256SUMS.txt'),
            user_visual_acceptance='User reported all 30 rooms passed on 2026-09-10; not a new automatic metric approval',
            schema='small_room30_weight_student_data_v1', records=self.records,
            evaluation_scope='All scenes are development; g2 checks noise repeats, not unseen scenes/motions',
            history_contract='Offline recorded motion prefix frames 0..7 at 20fps; no future frames or target IDs in forward',
            history_target='Geometric approach suitability heuristic, not observed human preference labels',
            raw_teacher_policy='Keep raw finite values, including out-of-range; direct multiplication; display clipping only')

    @lru_cache(maxsize=32)
    def geometry(self, scene):
        ds = self.data.scenes[scene]
        root = inside(self.data.root, ds['relation_labels']).parent
        manifest = json.loads((root / 'raw_scene_manifest.json').read_text())
        with np.load(inside(self.data.root, ds['supervision_only']), allow_pickle=False) as z:
            ids = z['instance_ids'].copy()
        with np.load(inside(self.data.root, ds['points_file']), allow_pickle=False) as z:
            points = z['points'].copy()
        instances = {str(i['stable_instance_id']): i for i in manifest['instances']}
        sample = next(s for s in self.data.samples if s['scene_id'] == scene and s['prompt_id'] == 'sit_anywhere')
        motions = [self.motions[i] for i in sample['source_motion_ids']]
        names = [str(m['target_instance_id']) for m in motions]
        masks = np.stack([ids == instances[name]['instance_id'] for name in names])
        if not masks.any(1).all() or len(set(names)) != len(names):
            raise ValueError('Missing or duplicate candidate support')
        anchors = np.asarray([[instances[n]['anchor_coordinate_local_xyz'][key] for key in ('x', 'z')] for n in names], np.float32)
        contacts = []
        histories = {}
        for motion in motions:
            path = inside(self.contact_root, motion['file'])
            if sha(path) != motion['sha256']:
                raise ValueError('Source contact/motion changed')
            with np.load(path, allow_pickle=False) as z:
                contacts.append(z['affordance'].copy())
                positions = z['motion_smpl22_20fps']
                histories[motion['id']] = observed_history_prefix_numpy(positions, np.arange(len(positions)) / 20.)
        relations = json.loads((root / 'relation_gt.json').read_text())['prompts']
        return points, ids, instances, names, masks, anchors, np.stack(contacts), histories, relations

    @lru_cache(maxsize=4)
    def teacher(self, filename):
        path = inside(self.summary_path.parent, 'cases/' + filename)
        if sha(path) != self.case_rows[filename]['sha256']:
            raise ValueError('Teacher cache changed during student run')
        return load_case(path)

    def example(self, record):
        points, ids, instances, names, masks, anchors, contacts, histories, relations = self.geometry(record['scene_id'])
        relation = next(r for r in relations if r['prompt_id'] == record['prompt_id'])
        selected = names.index(str(relation['selected_candidate_instance_id']))
        purpose = np.isin(ids, [instances[str(i)]['instance_id'] for i in relation['purpose_instance_ids']])
        if not purpose.any():
            raise ValueError('Missing purpose points')
        # Check relation annotations against exported coordinate anchors.
        context = np.asarray([[instances[str(i)]['anchor_coordinate_local_xyz'][k] for k in ('x', 'z')]
                              for i in relation['purpose_instance_ids']], np.float32)
        distances = np.linalg.norm(anchors[:, None] - context[None], axis=-1).min(1)
        if selected != int(distances.argmin()):
            raise ValueError('Relation selection does not match distance GT')
        history = histories[record['history_motion_id']].copy()
        scores = observed_scores(history, anchors)
        rs, hs, support = spatial_targets(contacts, masks, selected, scores, purpose)
        arrays = self.teacher(record['teacher_file'])
        sample = next(s for s in self.data.samples if s['id'] == record['sample_id'])
        with np.load(inside(self.data.root, sample['target_file']), allow_pickle=False) as z:
            expected_gt = z['affordance']
        if not np.array_equal(arrays['points'], points) or not np.array_equal(arrays['gt'], expected_gt):
            raise ValueError('Teacher/scene/GT point order mismatch')
        normalized = points.copy(); normalized[:, 3:] /= 255.
        inputs = dict(points=normalized, observed_history=history, teacher_a=arrays['adapted_raw'].copy())
        labels = dict(candidate_masks=masks.copy(), selected=np.asarray(selected, np.int64),
            history_scores=scores, purpose_mask=purpose.astype(np.float32),
            relation_spatial=rs, history_spatial=hs, contact_support=support.astype(np.float32))
        return inputs, labels
