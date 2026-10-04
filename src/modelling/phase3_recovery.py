"""Official-source recovery helpers for Phase 3A.3 (never forecasting)."""

from __future__ import annotations

import html
import re
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd


HCP_SEARCH = "https://www.hcp.ma/search/ipc/"
MONTHS = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}


def _fold(value: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(char)
    ).replace("’", "'")


def _reference_period(title: str) -> str | None:
    folded = _fold(title)
    annual = re.search(r"(?:annee|l'annee)\s+(20\d{2})", folded)
    if annual and "mois" not in folded:
        return f"{int(annual.group(1)):04d}-12"
    year = re.search(r"\b(20\d{2})\b", folded)
    if not year:
        return None
    for name, month in MONTHS.items():
        if re.search(rf"\b{name}\b", folded):
            return f"{int(year.group(1)):04d}-{month:02d}"
    return None


def recover_hcp_cpi_calendar(
    local_cpi_path: Path,
    output_path: Path,
    retrieval_date: str,
) -> pd.DataFrame:
    """Enumerate official HCP Actualite pages and match the local CPI periods."""

    records: dict[str, dict[str, str]] = {}

    def fetch(start: int) -> str:
        url = f"{HCP_SEARCH}?start_liste={start}"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        return urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")

    starts = list(range(0, 301, 20))
    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(fetch, starts))
    for start, page in zip(starts, pages):
        url = f"{HCP_SEARCH}?start_liste={start}"
        blocks = re.findall(
            r'<div class="result cel1.*?(?=<div class="result cel1|<div class="pager|$)',
            page, re.S,
        )
        for block in blocks:
            anchor = re.search(r'<h3.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
            dated = re.search(
                r'<div class="rubrique">(\d{2}/\d{2}/\d{4}).*?'
                r'<span class="rub">(.*?)</span>', block, re.S,
            )
            if not anchor or not dated or "Actualit" not in dated.group(2):
                continue
            title = " ".join(re.sub("<.*?>", " ", html.unescape(anchor.group(2))).split())
            if "prix à la consommation" not in title.lower():
                continue
            period = _reference_period(title)
            if period is None:
                continue
            publication = pd.to_datetime(dated.group(1), dayfirst=True).date().isoformat()
            candidate = {
                "reference_period": period,
                "publication_date": publication,
                "source_url": urljoin(HCP_SEARCH, html.unescape(anchor.group(1))),
                "source_document": title,
                "retrieval_date": retrieval_date,
                "timing_quality": "VERIFIED_DATE",
                "vintage_quality": "LATEST_REVISED",
                "status": "verified_first_release",
                "notes": "Date from contemporaneous HCP Actualite page; value from latest local extract.",
            }
            previous = records.get(period)
            if previous is None or candidate["publication_date"] < previous["publication_date"]:
                records[period] = candidate

    local = pd.read_csv(local_cpi_path)
    wanted = set(local["Mois"].str.replace("M", "-", regex=False).map(
        lambda value: "-".join((value.split("-")[0], value.split("-")[1].zfill(2)))
    ))
    recovered = pd.DataFrame([records[p] for p in sorted(wanted) if p in records])
    missing = sorted(wanted - set(recovered.reference_period))
    if missing:
        raise ValueError(f"official HCP release dates unresolved: {missing}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(output_path, index=False)
    return recovered


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    recover_hcp_cpi_calendar(
        root / "data/IPC/Indice des prix à la consommation (Mensuel) (Base 100 2017)_2026-06-25.csv",
        root / "data/metadata/hcp_cpi_release_calendar.csv",
        "2026-08-10",
    )
