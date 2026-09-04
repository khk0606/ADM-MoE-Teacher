# Publishing the Teacher-v10.2 reproduction assets

The source repository intentionally does not commit datasets, checkpoints, or
numeric experiment payloads. A fresh clone obtains the exact prerequisites from
the GitHub Release tag `teacher-v102-assets-v1`.

## Build the asset on the original WSL machine

Run from the AMDM repository root that contains the sealed Teacher-v10.1 result:

```bash
bash scripts/teacher_v102/package_assets.sh
```

The packager verifies the SHA-256 of every file bound by the Teacher-v10.1
summary. It includes only the data needed by the v10.2 run, converts bound JSON
paths from `/home/...` to repository-relative paths, and writes:

```text
dist/teacher-v102-assets-v1.tar.gz
dist/teacher-v102-assets-v1.tar.gz.sha256
```

The archive contains the author-created two-scene Teacher data, v5 replay data,
the original ADM/v5 checkpoints required by this experiment, and the sealed
v10/v10.1 evidence. It does not contain a Teacher-v10.2 checkpoint.

## Publish once with GitHub CLI

Confirm that every included third-party asset may be redistributed under its
original terms. Then authenticate `gh` and publish the immutable Release:

```bash
gh auth login
bash scripts/teacher_v102/publish_assets.sh
```

The publishing script refuses to replace an existing release. To publish a new
asset revision, change the asset tag/name in the source and create a new release
instead of overwriting v1.

## Fresh-clone verification

On a clean CUDA machine:

```bash
git clone https://github.com/khk0606/ADM-MoE-Teacher.git
cd ADM-MoE-Teacher
conda env create -f environment.teacher-v102.yml
conda activate afford
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
bash scripts/teacher_v102/bootstrap.sh
bash scripts/teacher_v102/reproduce.sh
bash scripts/teacher_v102/view.sh
```

`bootstrap.sh` downloads the Release asset, verifies the archive checksum,
refuses conflicting local files, and verifies every extracted payload and source
binding without training. `reproduce.sh` only proceeds when those assets are
already installed and then runs the v10.2 training and actual K=3 audit.
