# Source provenance

- `original-package.sha256`: original package manifest, retained byte-for-byte with its historical root paths.
- `package-tests.json`: original engineering tests, not final model performance.
- `layout.sha256`: manifest used by the reorganized launcher. Documentation/report paths are relocated; the launcher's checksum changes because it now reads this manifest. All 25 sealed Python/text-cache source entries retain their original digests.

No training manifest, model weight, numeric map, or machine approval field was rewritten. This is a repository-layout migration, not a new training run. Relative paths in the original test report describe the historical package.

Run from the repository root: `sha256sum -c docs/provenance/small_room30/layout.sha256`.
