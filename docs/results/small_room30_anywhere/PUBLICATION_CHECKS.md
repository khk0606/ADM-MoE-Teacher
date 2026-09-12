# Source publication checks — 2026-09-12

## Repository-layout follow-up

Execution guides and original package provenance now live under `docs/guides/` and `docs/provenance/small_room30/`. Upstream ADM entry points moved to `scripts/adm/`; launcher references and Hydra config paths were adjusted. Four dependency-free layout tests pass, including exact preservation of the 25 original sealed `prepare/` source/cache digests. The current launcher's layout manifest checks all 28 files. This does not replace model runtime testing.

## Initial publication checks

- Parsed all 439 Python source files in this publication tree successfully.
- Checked shell scripts with `bash -n`.
- Verified all 28 entries in `SMALL_ROOM30_ANYWHERE_SHA256SUMS.txt` without modifying their bound source.
- Checked current README/result-document local links: no broken links.
- Checked Git whitespace errors and obvious credential patterns in the new material.
- Five imported source files retain existing blank lines at EOF. These cosmetic warnings were left unchanged to preserve sealed source hashes.
- A local attempt to run the Anywhere unit tests stopped at import because the default macOS Python environment lacks NumPy. No model-quality or runtime pass is claimed from that attempt.
- No training, inference, or modification of server checkpoints was performed for publication.
- Owner-reported visual acceptance is recorded separately in `REVIEW_STATUS.md`; final numeric metrics and server payload hashes remain unpublished.

The source snapshot is not a fresh-install reproduction package. See `docs/DATA_AND_CHECKPOINTS.md` for required server assets and missing historical Teacher source seals.
