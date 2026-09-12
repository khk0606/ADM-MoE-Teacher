"""Add a real unconstrained Student prompt to every room, preserving all old labels."""
import copy
from functools import lru_cache
import numpy as np
from small_room30_competition_data import CompetitionData,POLICY,compete
from small_room30_student_data import TEXTS as LEGACY_TEXTS

ANYWHERE='sit_anywhere_v1'
TEXTS=dict(LEGACY_TEXTS,**{ANYWHERE:'Sit on something'})
RUN_POLICY=dict(POLICY,student_prompts=TEXTS,
    anywhere_relation='Both validated sittable candidates have R target 1; no purpose furniture constraint',
    anywhere_teacher='Same frozen1578 sit_anywhere A for both histories; no new Teacher generation')


class AnywhereData(CompetitionData):
    def __init__(self,dataset,teacher_summary):
        super().__init__(dataset,teacher_summary)
        self.legacy_binding=copy.deepcopy(self.binding)
        extra=[]
        for r in self.records:
            if r['prompt_id']==self.reference_prompt(r['scene_id']):
                extra.append(dict(r,prompt_id=ANYWHERE))
        self.records=list(self.records)+extra
        for r in extra:self.example(r)
        self.binding=copy.deepcopy(self.binding)
        self.binding.update(schema='small_room30_anywhere_data_v1',records=self.records,
            target_policy=RUN_POLICY,student_prompt_texts=TEXTS,
            anywhere_samples=len(extra),legacy_samples=len(self.legacy_binding['records']))
        if len(extra)!=180 or len(self.records)!=450:
            raise ValueError('Expected 30 rooms x 2 histories x 3 generations of added anywhere data')

    @lru_cache(maxsize=32)
    def reference_prompt(self,scene):
        # Only reuses the existing source/Teacher validation path. Its purpose
        # labels are explicitly replaced below; no scene-ID rule in forward.
        return self.geometry(scene)[-1][0]['prompt_id']

    @lru_cache(maxsize=180)
    def corrected_labels(self,scene,prompt,motion):
        if prompt!=ANYWHERE:return super().corrected_labels(scene,prompt,motion)
        l={k:v.copy() for k,v in super().corrected_labels(scene,self.reference_prompt(scene),motion).items()}
        r=np.ones(2,np.float32)
        score,q=compete(r,l['history_scores'])
        l.update(candidate_relation=r,purpose_classes=np.zeros(3,np.float32),
            purpose_mask=np.zeros_like(l['purpose_mask']),eligible=np.ones(2,bool),
            score_target=score,selection_target=q,
            relation_spatial=l['support_target'][:,:2].sum(-1).astype(np.float32),
            target_weight=(l['support_target'][:,:2]*q).sum(-1).astype(np.float32))
        return l

    def example(self,record):
        if record['prompt_id']!=ANYWHERE:return super().example(record)
        source=dict(record,prompt_id=self.reference_prompt(record['scene_id']))
        inputs,_=super().example(source)
        labels=self.corrected_labels(record['scene_id'],ANYWHERE,record['history_motion_id'])
        return inputs,{k:v.copy() for k,v in labels.items()}
