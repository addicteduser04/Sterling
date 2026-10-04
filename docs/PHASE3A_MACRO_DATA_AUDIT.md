# Phase 3A macro-financial data audit

> Superseded for recovery status by
> `docs/PHASE3A3_HISTORICAL_REAL_TIME_RECOVERY.md`. In particular, CPI release
> coverage is now 113/113 rather than 7/113.

## Decision

Phase 3A establishes the real-time data architecture but does **not** authorize
forecasting yet. `TAUX` is the target/endogenous BAM yield curve, not an
interbank or macro series. Of the audited domestic candidates, only the BAM
policy-rate history is accepted in the current checkout. `MACRO_DOMESTIC_V1`
is therefore **not frozen**: one sparse state variable is not the intended
domestic block, CPI timing is incomplete, and complete official money-market,
liquidity and Treasury-auction histories have not been preserved.

The machine-readable catalogue is `data/metadata/macro_catalogue.csv`; the
partial, verified CPI calendar is `data/metadata/hcp_cpi_release_calendar.csv`.
Running
`python -m src.modelling.phase3_data` deterministically writes the accepted
policy observations, checksum manifest, coverage/leakage tables, family audit,
economic mapping and information-block registry.

## Existing family audit

- **`TAUX`:** raw BAM yield-curve workbook plus an archived processed 23-tenor
  interpolated curve. It belongs only to `CURVE`. The legacy transformation
  renames observed tenors, linearly interpolates missing maturities on each
  date, uses endpoint carry for maturities outside observed knots, and forms
  monthly means. Values are percentage points. Its processed file is currently
  deleted in the working tree, so the audit records it without restoring it.
- **`IPC`:** 113 monthly headline index levels, 2017-01 through 2026-05,
  base 2017=100. The CSV/XLSX are latest extracts; they contain no release date
  or vintage. The legacy preprocessor incorrectly assigned the first day of the
  reference month. The strict parser now refuses all 113 values until every
  reference period has a verified first-release date.
- **`taux_directeur`:** 77 official decision rows, 2006-12-19 through
  2026-03-19. Source percentages are converted exactly once to decimal rates.
  The legacy monthly processor shifted decisions to the next month; the
  accepted event representation retains the actual decision date.
- **`taux_europe`:** 23 ECB AAA euro-area zero-coupon spot-rate tenors, 5,572
  daily rows from 2004-09-06 through 2026-06-24, expressed in percentage
  points. It is `MACRO_EXTERNAL_V1`, never part of the domestic block.
- **general processed data:** legacy CPI and policy monthly files use reference
  or shifted calendar dates, and the BAM/ECB merges are contemporaneous joins.
  None is an availability-safe domestic macro matrix.

## Source and timing findings

- **BAM policy rate:** accepted. The local official decision history is parsed
  as an event series; each decision becomes usable on its decision date. Since
  BAM forecast origins are date-only, this is conservative only for origins
  after the announcement; intraday applications must add timestamps.
- **HCP CPI:** rejected for now. The local monthly file contains reference
  periods and latest levels but no historical publication-date field or
  first-release vintage. HCP news pages contain dated releases, while the
  download catalogue visibly assigns later bulk-upload dates to groups of old
  observations. Seven recent first-release dates (2025-11 through 2026-05) are
  preserved, but 106 of 113 dates remain unresolved. No reference-month forward
  fill is allowed until the contemporaneous map is complete.
- **Interbank rates and liquidity:** rejected pending reproducible complete
  historical extraction and a verified daily/weekly release convention. BAM's
  interbank table confirms daily weighted-average rate, overnight volume and
  outstanding amount, but its CSV export is capped at the newest 1,000 records
  and ignores an offset parameter. That truncated export is validation evidence,
  not an accepted historical panel.
- **Treasury auctions:** rejected pending an official field dictionary,
  structured history, and result-publication timestamps. Auction date is not
  silently treated as publication date; bid-to-cover is not guessed.
- **EUR/MAD and USD/MAD:** promising but not accepted. BAM documents daily
  publication at 16:15 local time (14:00 during Ramadan) and possible same-day
  corrections. Date-only BAM origins should therefore use the prior published
  fixing unless their timestamp is demonstrably later.
- **Fiscal, money/credit, external, and oil:** excluded from the frozen initial
  block because releases are revised/low-frequency, their vintages are not in
  the checkout, or they are secondary to the domestic hypothesis.

## Leakage and vintage rules

Every observation has separate `reference_date`, `publication_date`, and
`available_from` fields. `available_from` may never precede publication. The
as-of join is backward-looking with exact matches permitted; it never
back-fills, interpolates, or merges on reference month. State variables may be
carried forward; maximum stale ages can invalidate them. Event flows may be
zero-filled only when a source definition proves zero is economically correct.
Revisions enter only when that vintage was available at the origin.

The CPI file is explicitly labelled latest/revised-value risk and is not
processed. `parse_cpi` fails closed when even one reference period lacks a
verified first-release date. This distinguishes a data-availability failure
from a negative forecast result: no claim about CPI predictability is possible.

## Missingness and sample impact

The accepted policy series is sparse by construction and is carried forward
after a decision, producing no missing state values after its first accepted
date. Exact coverage is written by the audit command. Other candidates have no
processed observations, so missingness and macro-limited origin counts are
reported as unavailable rather than fabricated. Strict matched-model origins
will be the intersection of complete curve and macro rows.

## Information-block design (separated, not pooled)

- `CURVE`: `TAUX`, NS/PCA factors and other endogenous curve features.
- `MACRO_DOMESTIC_V1`: policy rate, publication-aligned CPI, then verified
  interbank/liquidity/auction data and accepted FX. It is not frozen yet.
- `MACRO_EXTERNAL_V1`: the existing ECB curve; freeze only after domestic work.

Phase 3B must compare these blocks incrementally in that order and on strict
common origins. No Phase 3B forecast was run during this audit.

## Phase 2 reproduction and test status

The checked-in strict-common Phase 2 master table reproduces the report's
winners: DIRECT_RIDGE J+5 (0.6083% improvement), PCA_RIDGE J+10 (1.4601%), and
PCA_XGBOOST J+22 (3.6704%). These are point estimates, not robust winners.
Before Phase 3A changes, all 31 existing tests passed.

## Required acquisition next step

Preserve raw official BAM exports for daily interbank rates, liquidity
operations, Treasury auction results, and FX, including retrieval date, URL,
original filename, and checksum. Separately build the HCP CPI release calendar
from contemporaneous news releases and retain first-release values where
possible. Only then regenerate the audit, freeze accepted blocks, and implement
the matched Ridge models.
