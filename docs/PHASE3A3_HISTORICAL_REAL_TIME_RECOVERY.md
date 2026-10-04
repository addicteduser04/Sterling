# Phase 3A.3 — Historical Real-Time Data Recovery

## Status

**PHASE 3B BLOCKED — additional historical reconstruction required.**

`MACRO_DOMESTIC_V1` is not frozen and no forecasting was run. The exact
readiness gap is the missing official BAM interbank history between 2003-10-25
and 2023-11-09. This leaves only 24 of 45 existing monthly evaluation origins
with a complete three-series candidate block. The recovered liquidity and
Treasury-auction records cannot repair that limitation because their historical
public-release dates/timestamps are not established; they remain
`REFERENCE_DATE_ONLY` and are ineligible for forecasting.

This is a scientific-sample decision, not a performance decision. No RMSE, MAE,
DM statistic, or Phase 3B model was computed.

## Recovered evidence

### CPI

The HCP `Actualité` archive was enumerated independently of the download
catalogue. All 113 local reference months, 2017-01 through 2026-05, now map to a
direct contemporaneous HCP announcement and date. Coverage is 113/113 and
timing quality is `VERIFIED_DATE`.

The local values are nevertheless from a latest extract, not preserved
first-release files. Their vintage quality is therefore `LATEST_REVISED`, never
`FIRST_RELEASE`. The strict parser joins by release date and refuses an
incomplete calendar. Catalogue upload dates are not used.

### Interbank rate

BAM's official CSV endpoint was queried in both sort directions. It returned:

- oldest 1,000 observations: 2001-01-01 through 2003-10-24;
- newest 1,000 observations: 2023-11-10 through 2026-08-09.

The 7,322-day middle gap is explicit. The weighted-average rate is converted
once from percent to decimal; overnight volume and outstanding amounts are
retained in observation metadata. Same-day publication timing was unavailable,
so observations use the conservative next-calendar-origin rule and are classed
`CONSERVATIVE_NEXT_ORIGIN` / `UNKNOWN` vintage. A seven-day staleness cap
prevents the old endpoint from being carried across the gap.

### Liquidity and refinancing

The complete exposed BAM weekly archive was preserved: 264 PDFs (October 2019
through August 2026), with source URLs, retrieval dates, SHA-256 hashes and file
sizes. Conservative extraction produced 613 records, keeping distinct:

- total BAM interventions;
- seven-day advances outstanding;
- seven-day auction allotments;
- 24-hour advances outstanding.

The PDFs' week/operation dates are explicit, but the archive index supplies no
historical release date or timestamp. PDF creation metadata was not substituted
for public availability. All extracted records are consequently
`REFERENCE_DATE_ONLY` / `UNREVISED_ADMIN` and ineligible. One visibly malformed
official amount (`55 0000`) is excluded by the extractor rather than corrected
by assumption.

### Treasury auctions

The same weekly PDFs yielded 2,502 event-tenor rows from 2019-10-29 through
2026-08-04. There are 502 explicit accepted-amount observations and 501 accepted
yield observations. Auction date, maturity, repayments, accepted amount and
accepted yield are retained only when printed. Result-publication date, amount
demanded and settlement date remain null because the source did not provide
them in the extracted tables.

These events are `REFERENCE_DATE_ONLY` / `UNREVISED_ADMIN` and ineligible until
the corresponding dated Ministry/BAM result releases are linked.

## Candidate block accounting

The currently timing-eligible candidate contains three series/features:

1. BAM policy rate — `VERIFIED_DATE`, `UNREVISED_ADMIN`;
2. HCP CPI index — `VERIFIED_DATE`, `LATEST_REVISED`;
3. BAM weighted-average interbank rate — `CONSERVATIVE_NEXT_ORIGIN`, `UNKNOWN`.

Using the pre-existing Phase 2 origins and strict staleness rules:

| Frequency | Total origins | Complete origins | Reduction | First complete origin |
|---|---:|---:|---:|---|
| Monthly | 45 | 24 | 21 (46.7%) | 2023-11-30 |
| Weekly | 123 | 123 | 0 | 2024-01-05 |

Policy and CPI have no missing values at these origins. Interbank data are
missing at 21 monthly origins (46.7%). The weekly result is complete only because
the existing weekly evaluation starts in 2024; it does not create an adequate
historical training panel.

Staleness rules are: unbounded state carry for policy, 62 days for CPI, and 7
days for the daily interbank rate. Liquidity and auction records are not included
in this accounting because their timing quality is ineligible.

## Required next recovery

Readiness requires one of the following official-evidence outcomes:

1. recover the missing 2003-10-25 to 2023-11-09 BAM interbank observations with
   a defensible availability convention; or
2. establish release dates for a sufficiently long weekly BAM archive and use
   its interbank/liquidity series to form a broad, real-time monthly panel.

Treasury auction linkage remains desirable but is not the single blocking gap:
even fully timed 2019+ auctions would not restore the missing pre-2019 training
history. First-release CPI values are an unresolved vintage limitation, but the
release calendar itself is complete.
