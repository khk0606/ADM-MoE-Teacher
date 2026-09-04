# ADM-MoE-Teacher

ADM-MoE-Teacher extends the open-source Affordance Diffusion Model (ADM) with a Teacher-LoRA adaptation path and mixture-of-experts affordance reasoning.

## Installation and first ADM test

This guide assumes that the repository, Conda environment, PyTorch, and CUDA are already installed. Run every command from the repository root with the `afford` environment active.

### 1. Install `requirements.txt`

```bash
conda activate afford
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` installs PointOps, CLIP, PyTorch3D, Viser, and the remaining runtime packages. PointOps and PyTorch3D may compile CUDA extensions, so this step can take several minutes.

### 2. Check the installation

```bash
python -c "import sys, torch, numpy, clip, pytorch3d, viser; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('NumPy:', numpy.__version__); print('Viser:', viser.__version__)"
```

The reference CUDA environment prints values similar to:

```text
Python: 3.8.x
PyTorch: 1.12.0
CUDA available: True
CUDA runtime: 11.3
```

ADM inference requires `CUDA available: True`. If an import fails, reactivate the `afford` environment and rerun step 1 before continuing.

### 3. Prepare the teacher ADM test assets

The ADM source code is already included here, so do **not** clone Afford-Motion again. Open the official [Afford-Motion Data Preparation section](https://github.com/afford-motion/afford-motion#data-preparation), choose either the OneDrive or Baidu mirror, and download:

- **preprocessed data:** use only `custom`, `POINTTRANS_C_N8192_E300`, and the `Mean_Std_Cont_...0.8_fur.npz` file;
- **pre-trained models:** use only `CDM-Perceiver-ALL`.

Copy those files into this repository so the final layout is exactly:

```text
ADM-MoE-Teacher/
├── data/
│   ├── custom/
│   │   ├── anno.csv
│   │   └── points/*.npz
│   ├── POINTTRANS_C_N8192_E300/model.pth
│   └── Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz
└── outputs/CDM-Perceiver-ALL/ckpt/model300000.pt
```

The body model and the H3D, HUMANISE, HumanML3D, PROX, `data/eval`, and AMDM/CMDM assets are not needed for this test.

Check that all required files are present:

```bash
python -c "from pathlib import Path; required=['data/custom/anno.csv','data/POINTTRANS_C_N8192_E300/model.pth','data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz','outputs/CDM-Perceiver-ALL/ckpt/model300000.pt']; missing=[p for p in required if not Path(p).is_file()]; missing += [] if list(Path('data/custom/points').glob('*.npz')) else ['data/custom/points/*.npz']; print('READY' if not missing else 'MISSING:\n'+'\n'.join(missing)); raise SystemExit(bool(missing))"
```

Continue when the command prints `READY`. These external assets keep their original licenses and are intentionally not included in this repository.

See [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md) for integrity and licensing notes.

## 3-1 Generate and visualize Teacher-v10.2 affordance maps

Teacher-v10.2 applies dense per-instance supervision to Bed, normal Chair, and
High Chair instances in `room_0101` and `room_0102`. The model is trained with
equal macro weighting across the three object roles and evaluated using actual
500-step reverse-diffusion generation with `K=3` samples for the `watch` and
`write` prompts.

> **Research status:** Teacher-v10.2 is an experimental candidate. Its saved
> maps are visually useful, but it did not pass the strict all-three-object K=3
> gate. Therefore, no Teacher-v10.2 checkpoint was exported. The commands below
> reproduce and visualize the saved experimental rollout maps; they do not
> represent a validated final Teacher checkpoint.

### 3-2 Required Teacher assets

Prepare the following assets under the repository root:

```text
data/
├── history_affordance_relational_teacher_v7_hd/
├── history_affordance_relational_teacher_v9_all_sittable_v1/
│   ├── index.json
│   └── experiments/
├── history_affordance_v1/
│   ├── splits/chair23_bed2_whiteboard12_multistart24_v1.json
│   └── experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4/
│       ├── fewshot_cdm.pt
│       └── gate0a_report.json
└── Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz

outputs/
└── CDM-Perceiver-ALL/ckpt/model300000.pt
```

The relational datasets contain the two Unity point-cloud scenes, instance
labels, dense contact targets, and GT motion bindings used by this experiment.

### 3-3 Run Teacher-v10.2 training and actual K=3 generation

Teacher-v10.2 is a continuation experiment: its runner verifies the sealed
Teacher-v10.1 result before starting. To reproduce the complete research chain,
follow [Teacher experiment history](docs/TEACHER_EXPERIMENT_HISTORY.md). If the
required v10.1 evidence already exists, run this from the repository root:

```bash
bash teacher_lora_v102_dense_instance_patch/run_training.sh
```

The run performs supervised LoRA training and evaluates the shortlisted
`step_1000`, `step_800`, and `step_400` states using:

- two train scenes: `room_0101`, `room_0102`;
- two prompts: `watch`, `write`;
- three deterministic generations per prompt;
- 500 reverse-diffusion steps per generation.

The generated evidence is saved to:

```text
data/history_affordance_relational_teacher_v9_all_sittable_v1/
└── experiments/teacher_lora_v102/
    └── dense_instance_supervision_s20261031_v1/
        ├── summary.json
        └── dense_instance_supervision_maps.npz
```

`[DENSE_INSTANCE_SUPERVISION_FAIL]` is the recorded experimental gate result,
not a Python or CUDA execution error. The maps are retained for analysis even
though checkpoint export remains disabled.

### 3-4 Validate the saved result

```bash
python prepare/validate_relational_teacher_v102_dense_instance_supervision.py --summary data/history_affordance_relational_teacher_v9_all_sittable_v1/experiments/teacher_lora_v102/dense_instance_supervision_s20261031_v1/summary.json
```

The validator checks the bound datasets, saved map hash, array shapes, K=3
rollout metrics, v5 retention, and the no-checkpoint-on-failure policy.

### 3-5 Visualize the Teacher affordance maps with Viser

```bash
bash teacher_lora_v102_dense_instance_patch/run_viewer.sh
```

The viewer provides controls for:

- `room_0101` and `room_0102`;
- candidate steps `1000`, `800`, and `400`;
- generations `0`, `1`, and `2`;
- `watch` and `write` prompts;
- Bed, normal Chair, High Chair, or all verified objects;
- individual body-joint affordance channels.

The six panels are arranged as:

```text
Input RGB             **All-sittable GT       Frozen v5r4 Base
**Teacher-v10.2 result  Candidate − Base      |Candidate − GT|
```

Compare the top-middle ground-truth panel with the bottom-left Teacher-v10.2
panel.

The continuous surface is an XY interpolation for display only. Metrics and
gates use the original 8192 point-aligned values.

### * 4. 5. is for Original Affordance ADM. Can skip
### 4. Run the original ADM test

After placing the official novel-evaluation data and `CDM-Perceiver-ALL` checkpoint, run:

```bash
bash scripts/novel_contact/test.sh outputs/CDM-Perceiver-ALL 2023
```

The script loads `outputs/CDM-Perceiver-ALL/ckpt/model300000.pt`, reads the language prompts and 8192-point scenes in `data/custom/`, and runs 500 reverse-diffusion steps with seed `2023`. It generates a six-channel contact-distance map for the pelvis, left/right foot, neck, and left/right wrist. Results are written under a new time-stamped directory:

```text
outputs/CDM-Perceiver-ALL/eval/test-MMDD-HHMMSS/
├── custom/
│   └── pred_contact/
│       ├── 00000.npy
│       ├── 00001.npy
│       └── ...
├── metrics.txt
└── test.log
```

Each prediction file stores contact **distance**, not an image. Its shape is `[K, 8192, 6]`: `K=30` for the first K-sampled cases selected by the current test script and `K=1` for an ordinary single-sample case. Smaller distance means stronger contact/affordance. Because `scripts/novel_contact/test.sh` currently sets `eval_metrics=[]`, `metrics.txt` is expected to be empty; `custom/pred_contact/*.npy` and `test.log` are the useful outputs.

If the command reports `No checkpoint found`, confirm that this exact file exists:

```text
outputs/CDM-Perceiver-ALL/ckpt/model300000.pt
```

### 5. Visualize the original ADM result with Viser

Use the exact time-stamped folder created by step 4. For example, if the test created `test-0904-183000`, run this as one line:

```bash
python visualize_adm_affordance_viser.py --eval-dir outputs/CDM-Perceiver-ALL/eval/test-0904-183000 --data-root data --host 0.0.0.0 --port 8080
```

The Viser page shows two point-aligned panels:

- **Input 3D Scene (RGB):** the original scene in `data/custom/points/`;
- **ADM Affordance Map:** the selected prediction converted from distance to physical affordance with the same `sigma=0.8` used by the test.

Use **Sample / scene / prompt** to change the input case, **Generation** to inspect individual K samples, and **Affordance channel** to switch between `any_joint` and the six body-joint channels. The fixed color scale is purple `0` to red `1`. `RGB blend` changes only the display and never modifies the saved `.npy` result.


## Project overview

The goal is to generate a history-conditioned affordance map rather than treating motion generation itself as the primary contribution.

```text
text + 3D scene
        |
        v
frozen ADM + LoRA Teacher  ----->  base affordance map A
                                       |
text + 3D scene + past history         |
        |                              |
        v                              v
      MoE-IIW  --------------------> weighted map A^w
                                       |
                                       v
                               frozen CMDM/AMDM
                                       |
                                       v
                                  motion M
```

The Teacher learns general action-compatible regions from text, scene point clouds, and dense motion-contact supervision. Instance identity, object relations, purpose-object distance, and motion identity are supervision metadata and are excluded from the Teacher forward input.

The MoE stage conditions and selects among valid candidates using history. A planned relation-aware extension can additionally use purpose compatibility, such as choosing a sittable object appropriate for watching or writing.

## Repository scope

This source snapshot contains:

- the ADM/CMDM base implementation;
- Base Teacher and LoRA utilities;
- MoE-IIW model, routing, training, evaluation, and contract tests;
- Teacher-v7 data staging, dense-contact, LoRA, pilot, K=1, K=3, and full-train evaluation source;
- the complete Teacher-v9 through Teacher-v10.3.1 experiment packages,
  contracts, objectives, validators, summarizers, and Viser viewers;
- the Unity motion-audit source used before dense-contact generation.

It intentionally excludes upstream checkpoints, body models, generated numeric
maps, machine-local experiment summaries, videos, and local metadata. The
author-created relational scene/GT assets may be distributed separately with a
public result bundle.

See [Repository scope](docs/REPOSITORY_SCOPE.md) for the exact inclusion and exclusion policy.

## Source layout

```text
configs/       Hydra configurations for ADM and CMDM
datasets/      dataset loaders and transforms
diffusion/     diffusion implementation
models/        ADM, CMDM, LoRA, MoE-IIW, and routing modules
prepare/       data preparation, training, evaluation, validators, and tests
scripts/       shell entry points for ADM/CMDM experiments
teacher_lora_* versioned Teacher-v9 through Teacher-v10.3.1 research packages
unity/         Unity-side physical motion audit source
utils/         shared utilities
docs/          Teacher runbooks and repository-scope documentation
```

## Current research status

Teacher-v10.2 is retained as the current visual comparison candidate. Its dense
per-instance supervision produces strong pooled recall on most labelled
instances, but every scene/prompt combination recorded `0/3` generations that
simultaneously passed all three absolute-object checks. Its step-1000 v5 fixed
probe also changed by `+6.321%`, beyond the locked 5% retention limit.

Accordingly:

- Teacher-v10.2 status is **FAIL** under the strict actual-K3 gate;
- `selected_step` is `None`;
- no Teacher-v10.2 LoRA checkpoint was written;
- the saved maps and screenshots are experimental evidence, not a validated
  final Teacher release.

Later Teacher-v10.3/v10.3.1 on-policy experiments are included as research
history. They do not supersede this status or authorize a final checkpoint.
See [Teacher experiment history](docs/TEACHER_EXPERIMENT_HISTORY.md) and the
[Teacher-v10.2 public result summary](docs/results/teacher_v102/README.md).

## Upstream attribution

The base implementation is derived from the official code for *Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance* (CVPR 2024):

- [Official code](https://github.com/afford-motion/afford-motion)
- [Project page](https://afford-motion.github.io/)
- [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Move_as_You_Say_Interact_as_You_Can_Language-guided_Human_CVPR_2024_paper.html)

```bibtex
@inproceedings{wang2024move,
  title={Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance},
  author={Wang, Zan and Chen, Yixin and Jia, Baoxiong and Li, Puhao and Zhang, Jinlu and Zhang, Jingze and Liu, Tengyu and Zhu, Yixin and Liang, Wei and Huang, Siyuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## License

See [LICENSE](LICENSE). Dataset, pretrained-model, Unity-asset, and body-model licenses are separate and those files are not included in this repository.
