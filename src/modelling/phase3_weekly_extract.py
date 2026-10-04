"""Conservative extraction of structured BAM weekly indicator PDFs."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


def _number(value: str) -> float | None:
    value = value.strip()
    if value in {"", ".", "-"}:
        return None
    return float(value.replace(" ", "").replace(",", "."))


def extract_structured_weekly(text: str, source_file: str) -> tuple[list[dict], list[dict]]:
    """Extract only explicitly labelled liquidity and auction fields.

    Layouts that do not match are skipped; no value or date is inferred.
    Publication timing remains unresolved because the BAM archive page does not
    expose historical release dates.
    """

    liquidity: list[dict] = []
    auctions: list[dict] = []
    week_matches = re.findall(
        r"Semaine\s+Semaine\s+du\s+(\d{2}-\d{2}-\d{2}).*?"
        r"du\s+(\d{2}-\d{2}-\d{2}).*?au\s+\d{2}-\d{2}-\d{2}\s+au\s+(\d{2}-\d{2}-\d{2})",
        text, re.S | re.I,
    )
    current_end = pd.to_datetime(week_matches[0][2], dayfirst=True) if week_matches else pd.NaT
    section = re.search(
        r"Interventions\*?\s+de Bank Al-Maghrib(.*?)(?:MARCHE DES ADJUDICATIONS|INFLATION)",
        text, re.S | re.I,
    )
    if section and pd.notna(current_end):
        body = section.group(1)
        definitions = {
            "bam_liquidity_total_interventions": r"^\s*Interventions de BAM\s+",
            "bam_liquidity_7d_advances_stock": r"^\s*-?Avances à 7 jours\s+",
            "bam_liquidity_24h_advances_stock": r"^\s*-?Avances à 24 heures\s+",
            "bam_liquidity_deposit_facility_stock": r"^\s*-?Facilités de dépôt à 24 heures\s+",
        }
        for series_id, pattern in definitions.items():
            match = re.search(pattern + r"([^\n]+)$", body, re.M | re.I)
            columns = re.split(r"\s{2,}", match.group(1).strip()) if match else []
            value = _number(columns[-1]) if len(columns) >= 2 else None
            if value is not None:
                liquidity.append({
                    "series_id": series_id, "reference_date": current_end.date().isoformat(),
                    "value": value, "unit": "MAD_millions", "source_file": source_file,
                    "timing_quality": "REFERENCE_DATE_ONLY", "vintage_quality": "UNREVISED_ADMIN",
                })
        allotment = re.search(
            r"Résultats des avances à 7 jours.*?du\s+(\d{2}/\d{2}/\d{4}).*?"
            r"Montant servi\s+([\d ]+)", body, re.S | re.I,
        )
        allotment_value = _number(allotment.group(2)) if allotment else None
        if allotment and allotment_value is not None and allotment_value <= 200_000:
            liquidity.append({
                "series_id": "bam_liquidity_7d_allotment", "reference_date": pd.to_datetime(
                    allotment.group(1), dayfirst=True
                ).date().isoformat(), "value": allotment_value,
                "unit": "MAD_millions", "source_file": source_file,
                "timing_quality": "REFERENCE_DATE_ONLY", "vintage_quality": "UNREVISED_ADMIN",
            })

    auction_section = re.search(
        r"MARCHE DES ADJUDICATIONS(.*?)(?:INFLATION|TAUX D.INTERET|$)", text, re.S | re.I
    )
    if auction_section:
        body = auction_section.group(1)
        auction_date = re.search(r"Adjudications? au\s+(\d{2}-\d{2}-\d{2})", body, re.I)
        if auction_date:
            date = pd.to_datetime(auction_date.group(1), dayfirst=True).date().isoformat()
            maturity_pattern = r"(45 jours|13 semaines|26 semaines|52 semaines|2 ans|5 ans|10 ans|15 ans|20 ans|30 ans)"
            for line in body.splitlines():
                maturity = re.search(maturity_pattern, line, re.I)
                if not maturity:
                    continue
                fields = re.split(r"\s{2,}", line[maturity.end():].strip())
                if len(fields) < 3:
                    continue
                auctions.append({
                    "auction_date": date, "result_publication_date": None,
                    "maturity": maturity.group(1), "amount_repaid_mad_millions": _number(fields[0]),
                    "amount_accepted_mad_millions": _number(fields[1]),
                    "accepted_yield_percent": _number(fields[2]), "amount_demanded_mad_millions": None,
                    "settlement_date": None, "source_file": source_file,
                    "timing_quality": "REFERENCE_DATE_ONLY", "vintage_quality": "UNREVISED_ADMIN",
                })
    return liquidity, auctions


def extract_archive(raw_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    liquidity: list[dict] = []
    auctions: list[dict] = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        with tempfile.NamedTemporaryFile(suffix=".txt") as temporary:
            subprocess.run(["pdftotext", "-layout", str(pdf), temporary.name], check=True)
            text = Path(temporary.name).read_text(encoding="utf-8", errors="replace")
        found_liquidity, found_auctions = extract_structured_weekly(text, pdf.name)
        liquidity.extend(found_liquidity)
        auctions.extend(found_auctions)
    output_dir.mkdir(parents=True, exist_ok=True)
    liquidity_frame = pd.DataFrame(liquidity).drop_duplicates()
    auction_frame = pd.DataFrame(auctions).drop_duplicates()
    liquidity_frame.to_csv(output_dir / "bam_weekly_liquidity_reference_only.csv", index=False)
    auction_frame.to_csv(output_dir / "treasury_auctions_reference_only.csv", index=False)
    return liquidity_frame, auction_frame
