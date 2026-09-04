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

### 3. Prepare the ADM test assets

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

### 4. Run the original ADM test

After placing the official novel-evaluation data and `CDM-Perceiver-ALL` checkpoint, run:

```bash
conda activate afford
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

### 5. Visualize the ADM result with Viser

Use the exact time-stamped folder created by step 4. For example, if the test created `test-0904-183000`, run this as one line:

```bash
python visualize_adm_affordance_viser.py --eval-dir outputs/CDM-Perceiver-ALL/eval/test-0904-183000 --data-root data --host 0.0.0.0 --port 8080
```

The terminal prints `[OPEN] http://localhost:8080`. Open that address in a browser. If Ubuntu is running on a remote machine, forward port `8080` to the local computer first.

The Viser page shows two point-aligned panels:

- **Input 3D Scene (RGB):** the original scene in `data/custom/points/`;
- **ADM Affordance Map:** the selected prediction converted from distance to physical affordance with the same `sigma=0.8` used by the test.

Use **Sample / scene / prompt** to change the input case, **Generation** to inspect individual K samples, and **Affordance channel** to switch between `any_joint` and the six body-joint channels. The fixed color scale is purple `0` to red `1`. `RGB blend` changes only the display and never modifies the saved `.npy` result.

Stop the viewer with `Ctrl+C`.

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
