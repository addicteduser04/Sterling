# Repository maintenance

The public repository keeps modelling source, runtime dependencies, research reports, selected figures, CSV inputs, and data provenance.

Local tests (`tests/`) and scratch directories are ignored at the owner's request. The existing tests contain regression checks for rate units, leakage controls, model reconstruction, and horizon handling; they are preserved locally. To run them where available, install pytest and run `python -m pytest -q` from the repository root.

All `outputs_*` directories are generated experiment artifacts and ignored. Two saved experiment reports and two Phase 6 figures are retained under `docs/`. Research reports describe the original runs; references to output tables refer to locally regenerated artifacts.

Bulky weekly PDFs, CPI spreadsheets, the raw BAM workbook, and derived data panels are ignored and preserved locally. Source CSVs and manifests remain tracked because the pipelines rely on them. Ignore rules also cover Python environments, caches, local credentials, editors, and temporary files.

Ignoring a file does not remove it from earlier Git history. This cleanup removes ignored files from the current Git index without deleting local copies or rewriting history.
