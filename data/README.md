# Research data

CSV inputs are retained so the modelling code remains inspectable and reproducible. `metadata/` records source URLs, checksums, acquisition details, and release-date evidence.

- `combined/bam_ecb_2004.csv`: aligned legacy BAM/ECB panel; both rate blocks are in percentage points. Load with explicit percent units.
- `taux_europe/`: ECB source exports used by the cross-market loaders.
- `taux_us/raw/`: annual U.S. Treasury CSV exports used by the cross-market loaders.
- `IPC/`, `taux_directeur/`, and `interbank/raw/`: domestic macro source CSVs.
- `bam_weekly/raw/source_urls.csv`: provenance for the locally retained weekly PDF archive.

Derived `processed/` directories and bulky PDF, XLSX, and XLSM archives are ignored. The weekly PDF extractor and raw spreadsheet audits require those local archives; a fresh clone does not include them. Use the source manifests to identify the required files. Historical macro availability and vintage gaps are documented in the Phase 3 reports; reference-date-only observations are not accepted as real-time features.
