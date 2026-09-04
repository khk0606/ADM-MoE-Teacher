# ADM-MoE-Teacher

ADM-MoE-Teacher extends the open-source Affordance Diffusion Model (ADM) with a Teacher-LoRA adaptation path and mixture-of-experts affordance reasoning.

## Installation and verification

This guide assumes that the repository, Conda environment, PyTorch, and CUDA are already installed. Run every command from the repository root with the `afford` environment active.

### 1. Install repository dependencies

```bash
conda activate afford
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` installs PointOps, CLIP, PyTorch3D, and the remaining runtime packages. PointOps and PyTorch3D may compile CUDA extensions, so this step can take several minutes.

### 2. Check the installation

```bash
python -c "import sys, torch, numpy; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('NumPy:', numpy.__version__)"
```

The reference CUDA environment prints values similar to:

```text
Python: 3.8.x
PyTorch: 1.12.0
CUDA available: True
CUDA runtime: 11.3
```

`CUDA available: False` is acceptable only for the data-free CPU tests below. Full ADM/Teacher inference and training require an NVIDIA GPU and CUDA.

### 3. Run the data-free CPU tests

These tests use synthetic temporary data. They do not download datasets, load checkpoints, or run an optimizer.

```bash
python prepare/test_base_teacher_contract.py
python prepare/test_relational_teacher_v7_hd_preflight_contract.py
python prepare/test_relational_teacher_v7_hd_k3_contract.py
```

The final output must contain:

```text
Ran 55 tests
OK
[PASS] Teacher-v7 High-Desk probe selection and tamper guard
[PASS] only sealed hc_hd train rows can enter CUDA preflight
[PASS] canonical binding arithmetic is order invariant
[PASS] v7 K=3 pooled, per-generation and High-Desk gates
[PASS] multi-seed High-Desk/invariance tamper guards
```

If these tests pass, the Python environment and source-level Teacher contracts are working. This does not yet test a real checkpoint or scene.

### 4. Prepare external data and checkpoints

Datasets, body models, checkpoints, generated maps, and experiment outputs are not stored in this repository.

The base ADM/CMDM implementation comes from the official [afford-motion repository](https://github.com/afford-motion/afford-motion). Its README provides the official OneDrive/Baidu links for preprocessed datasets and pretrained models.

Download the upstream assets and preserve their internal directory names. The minimum expected layout for the novel-scene ADM is:

```text
ADM-MoE-Teacher/
├── body_models/
│   └── smplx/
│       └── SMPLX_NEUTRAL.npz
├── data/
│   ├── custom/
│   ├── eval/
│   ├── H3D/
│   ├── HUMANISE/
│   ├── HumanML3D/
│   ├── PROX/
│   ├── POINTTRANS_C_N8192_E300/
│   └── Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz
└── outputs/
    └── CDM-Perceiver-ALL/
        └── ckpt/
            └── model300000.pt
```

The Teacher-v7 CUDA workflow additionally expects:

```text
data/
├── history_affordance_relational_teacher_v7_hd/
│   └── index.json
└── history_affordance_v1/
    ├── splits/
    └── experiments/
        └── fewshot_cdm_chair23_bed2_whiteboard12_v5r4/
            ├── fewshot_cdm.pt
            └── gate0a_report.json
```

The relational Teacher dataset and v5r4 checkpoint are project-specific assets and are not currently distributed by this repository. Installation and CPU contract testing work without them, but the CUDA Teacher preflight requires them.

See [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md) for integrity and licensing notes.

### 5. Test the original ADM checkpoint

After placing the official novel-evaluation data and `CDM-Perceiver-ALL` checkpoint, run:

```bash
conda activate afford
bash scripts/novel_contact/test.sh outputs/CDM-Perceiver-ALL 2023
```

The script runs 500 diffusion steps with seed `2023`. Results are written under a time-stamped directory similar to:

```text
outputs/CDM-Perceiver-ALL/eval/test-MMDD-HHMMSS/
```

If the command reports `No checkpoint found`, confirm that this exact file exists:

```text
outputs/CDM-Perceiver-ALL/ckpt/model300000.pt
```

### 6. Validate the Teacher-v7 dataset

Run this only after the Teacher-v7 assets have been placed correctly:

```bash
python prepare/validate_relational_teacher_v7_hd_dataset.py --dataset-root data/history_affordance_relational_teacher_v7_hd --index data/history_affordance_relational_teacher_v7_hd/index.json
```

Do not continue if validation reports a missing file, hash mismatch, scene mismatch, or failed contract.

### 7. Run the Teacher-LoRA CUDA preflight

The preflight performs no optimizer update and does not save a candidate checkpoint. It verifies dataset/checkpoint bindings, installs a fresh zero-output LoRA, and confirms that valid gradients reach only the LoRA parameters.

Create a new output directory name for every run:

```bash
DATA=data/history_affordance_relational_teacher_v7_hd
V5=data/history_affordance_v1/experiments/fewshot_cdm_chair23_bed2_whiteboard12_v5r4
OUT=$DATA/experiments/teacher_lora_v7_hd/preflight_local_v1
```

Run the preflight as one complete command:

```bash
python -u prepare/preflight_relational_teacher_v7_hd_lora.py --dataset-root "$DATA" --index "$DATA/index.json" --stats-file data/Mean_Std_Cont_HumanML3D_HUMANISE_PROX_contact_cont_joints_0.8_fur.npz --original-checkpoint outputs/CDM-Perceiver-ALL/ckpt/model300000.pt --v5-checkpoint "$V5/fewshot_cdm.pt" --v5-evidence-report "$V5/gate0a_report.json" --report "$OUT/preflight.json" --diffusion-steps 500 --lora-rank 4 --lora-alpha 8 --seed 20260908 --device cuda:0
```

Validate the generated report:

```bash
python prepare/validate_relational_teacher_v7_hd_lora_preflight.py --report "$OUT/preflight.json"
```

Proceed only if both commands print their PASS conclusions. The later sealed training and evaluation commands are under `docs/`.

### 8. Visualize a frozen Base-Teacher cache

If a generated Base-Teacher cache index is available, render point-aligned PNG and PLY artifacts without new GPU inference:

```bash
python visualize_frozen_base_affordance.py --index /path/to/base_teacher_cache_index.json --dataset-root /path/to/history_affordance_dataset --output-dir outputs/base_teacher_visualization
```

Use `--skip-ply` when only PNG heatmaps are needed.

### Common execution problems

#### `torch.cuda.is_available()` is `False`

Check that `nvidia-smi` works in the same Ubuntu/WSL terminal. If it does not, fix the NVIDIA driver or WSL GPU configuration first.

#### PointOps or PyTorch3D fails to compile

```bash
gcc --version
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The system CUDA compiler used for extensions must be compatible with the PyTorch 1.12/CUDA 11.3 environment.

#### `ModuleNotFoundError`

```bash
conda activate afford
pwd
python -m pip check
```

Confirm that the `afford` environment is active and the terminal is at the repository root.

#### The terminal shows `>` instead of running the command

The shell is waiting for an unfinished quote or line continuation. Press `Ctrl+C`, then paste the complete command again. Long runnable commands in this README are intentionally written on one line.

#### The output directory already exists

Teacher scripts intentionally avoid overwriting evidence. Use a new versioned path such as `preflight_local_v2` instead of deleting an accepted result.

## Project overview

The goal is to generate a history-conditioned affordance map rather than treating motion generation itself as the primary contribution.

```text
text + 3D scene
        |
        v
frozen ADM + LoRA Teacher  ----->  base affordance map A
                                      |
text + 3D scene + past history        |
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
- the Unity motion-audit source used before dense-contact generation.

It intentionally excludes datasets, raw motion TXT files, Unity scene exports, experiment summaries, generated numeric affordance maps, checkpoints, videos, local metadata, and superseded patch archives.

See [Repository scope](docs/REPOSITORY_SCOPE.md) for the exact inclusion and exclusion policy.

## Source layout

```text
configs/       Hydra configurations for ADM and CMDM
datasets/      dataset loaders and transforms
diffusion/     diffusion implementation
models/        ADM, CMDM, LoRA, MoE-IIW, and routing modules
prepare/       data preparation, training, evaluation, validators, and tests
scripts/       shell entry points for ADM/CMDM experiments
unity/         Unity-side physical motion audit source
utils/         shared utilities
docs/          Teacher runbooks and repository-scope documentation
```

## Validation-first Teacher workflow

The current Teacher path is deliberately gated:

1. stage and validate scene/motion bindings;
2. generate and independently validate dense contact;
3. run LoRA CUDA zero-initialization and gradient preflight;
4. run the fresh 12-update recovery calibration;
5. run the bounded 60-update pilot;
6. evaluate only the shortlisted checkpoint with paired K=1 and K=3 train-only canaries;
7. run full train-only K=3 before development evaluation.

Every gate fails closed. A failed validation does not authorize the following training or evaluation stage.

## Current verified status

The last sealed milestone represented by this source snapshot is the Teacher-v7 selected update-12 train-only K=3 canary:

- pooled target MAE relative change: `-1.0502%`;
- pooled case win rate: `81.25%`;
- failed checks: none.

These values are a status statement only; generated maps and evaluation payloads are not committed. Development room `room_0201` was not read by this train-only gate. This repository does not claim a completed development result, held-out result, or independent paper-test result.

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
