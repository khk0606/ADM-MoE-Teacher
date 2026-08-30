# ADM-MoE-Teacher

ADM-MoE-Teacher extends the open-source Affordance Diffusion Model (ADM) pipeline with a teacher adaptation path and mixture-of-experts affordance reasoning.

The research goal is to generate a history-conditioned affordance map, rather than treating motion generation itself as the primary contribution. The intended pipeline is:

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
                               frozen CMDM
                                      |
                                      v
                                  motion M
```

The Teacher learns general action-compatible regions from text, scene point clouds, and dense motion-contact supervision. Instance identity, object relations, and purpose-object distance are supervision metadata and are excluded from the Teacher forward input. The MoE stage is responsible for conditioning and selecting among valid candidates using history and, in the planned relation-aware extension, purpose compatibility such as choosing a sittable object appropriate for watching or writing.

## Repository scope

This is a source-only research snapshot. It contains:

- the ADM/CMDM base implementation;
- Base Teacher and LoRA utilities;
- MoE-IIW model, routing, training, evaluation, and contract tests;
- the current Teacher-v7 data-staging, dense-contact, LoRA, pilot, K=1, K=3, and full-train evaluation code;
- the Unity motion-audit source used before dense-contact generation.

It intentionally excludes datasets, raw motion TXT files, Unity scene exports, experiment summaries, generated affordance maps, checkpoints, videos, local metadata, and superseded patch archives. See [Repository scope](docs/REPOSITORY_SCOPE.md) and [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md).

## Verified status

The last sealed milestone represented by this snapshot is the Teacher-v7 selected update-12 train-only K=3 canary:

- pooled target MAE relative change: `-1.0502%`;
- pooled case win rate: `81.25%`;
- failed checks: none.

These numbers are included only as a status statement; the generated maps and evaluation payloads are not committed. Development room `room_0201` was not read by this train-only gate. Full train-only K=3, development evaluation, and an independent paper test are not claimed as completed here.

## Installation

The original environment used Python 3.8, PyTorch 1.12, and CUDA 11.3.

```bash
conda create -n afford python=3.8
conda activate afford
conda install pytorch==1.12.0 torchvision==0.13.0 \
  torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Some 3D operators require a CUDA compiler compatible with the selected PyTorch build.

## Main source layout

```text
configs/       ADM and CMDM configurations
datasets/      dataset loaders and transforms
diffusion/     diffusion implementation
models/        ADM, CMDM, MoE-IIW, and routing modules
prepare/       data preparation, training, evaluation, validators, and tests
scripts/       original ADM/CMDM entry scripts
unity/         Unity-side physical motion audit source
docs/          current runbooks and repository-scope documentation
```

## Validation-first workflow

The current Teacher path is deliberately gated:

1. stage and validate scene/motion bindings;
2. generate and recompute dense contact;
3. run LoRA CUDA zero-initialization and gradient preflight;
4. run the fresh 12-update recovery calibration;
5. run the bounded 60-update pilot;
6. evaluate only the shortlisted checkpoint with paired K=1 and K=3 train-only canaries;
7. run full train-only K=3 before any development evaluation.

The runnable commands are documented in `docs/`. They require external data and checkpoints described in [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md).

## Upstream attribution

The base implementation is derived from the official code for *Move as You Say, Interact as You Can: Language-guided Human Motion Generation with Scene Affordance* (CVPR 2024). Its MIT license is retained in [LICENSE](LICENSE).

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
