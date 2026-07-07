import re
from pathlib import Path
import pandas as pd
import unicodedata


YIELDS = ['3M','6M','1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y','11Y','12Y','13Y','14Y','15Y','16Y','17Y','18Y','19Y','20Y','30Y']
RAW_YIELDS = {
    '13 semaines': '3M', '26 semaines': '6M', '52 semaines': '1Y',
    '2 ans': '2Y', '5 ans': '5Y', '10 ans': '10Y', '15 ans': '15Y',
    '20 ans': '20Y', '30 ans': '30Y'
}


def normalize_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r'\s+', ' ', s)
    return s


def find_date_in_parents(path: Path):
    for p in path.parents:
        m = re.match(r"^(\d{8})$", p.name)
        if m:
            return m.group(1)
    return None


def parse_file_to_yield_map(p: Path):
    # Returns dict mapping standard YIELDS -> float or None
    try:
        df = pd.read_excel(p, header=0, engine="openpyxl")
    except Exception:
        try:
            df = pd.read_excel(p, header=None, engine="openpyxl")
        except Exception:
            return None

    # Normalize column names
    cols = [normalize_label(c) for c in df.columns]

    # Case A: tenor column exists (Tenor/Tensor)
    tenor_col_idx = None
    for i, c in enumerate(cols):
        if c in ("tenor", "tensor", "tenor\n"):
            tenor_col_idx = i
            break

    result = {y: None for y in YIELDS}

    if tenor_col_idx is not None:
        tenor_col = df.columns[tenor_col_idx]
        # locate a column with numeric values (first one)
        value_col = None
        for c in df.columns:
            if c == tenor_col:
                continue
            series = pd.to_numeric(df[c], errors='coerce')
            if series.dropna().size > 0:
                value_col = c
                break
        if value_col is None:
            return result

        for raw, std in RAW_YIELDS.items():
            # find matching tenor in tenor_col
            matches = df[tenor_col].astype(str).apply(normalize_label) == normalize_label(raw)
            if matches.any():
                val = pd.to_numeric(df.loc[matches, value_col].iloc[0], errors='coerce')
                result[std] = None if pd.isna(val) else float(val)

        return result

    # Case B: tenor labels appear in dataframe values or column names
    # Check column names first
    for c in df.columns:
        label_norm = normalize_label(c)
        for raw, std in RAW_YIELDS.items():
            if label_norm == normalize_label(raw):
                # take first non-null value in that column
                val = pd.to_numeric(df[c].dropna().iloc[0], errors='coerce') if df[c].dropna().size>0 else None
                result[std] = None if pd.isna(val) else float(val)

    # Check cell values: find a row where tenor labels exist horizontally
    for idx, row in df.iterrows():
        row_norm = [normalize_label(x) for x in row.tolist()]
        for j, cell in enumerate(row_norm):
            for raw, std in RAW_YIELDS.items():
                if cell == normalize_label(raw):
                    # value might be in same column in next row
                    try:
                        val = pd.to_numeric(df.iloc[idx+1, j], errors='coerce')
                    except Exception:
                        val = None
                    result[std] = None if pd.isna(val) else float(val)

    # As a last resort, take first numeric row and map by position to YIELDS
    flat_vals = None
    for idx, row in df.iterrows():
        numeric = pd.to_numeric(row, errors='coerce')
        if numeric.dropna().size > 0:
            flat_vals = [None if pd.isna(x) else float(x) for x in numeric.tolist()]
            break

    if flat_vals is not None:
        for i, y in enumerate(YIELDS):
            if i < len(flat_vals):
                result[y] = flat_vals[i]

    return result


def merge_taux(taux_dir: Path, out_csv: Path):
    records = []
    files_processed = 0
    for p in taux_dir.rglob("*.xlsx"):
        if p.name.startswith("~$"):
            continue
        date = find_date_in_parents(p)
        if date is None:
            m = re.search(r"(\d{8})", p.name)
            date = m.group(1) if m else p.stem

        mapping = parse_file_to_yield_map(p)
        if mapping is None:
            continue
        mapping["date"] = pd.to_datetime(date, format="%Y%m%d", errors="coerce")
        records.append(mapping)
        files_processed += 1

    if files_processed == 0:
        print("No Excel files found/processed in", taux_dir)
        return

    out_df = pd.DataFrame.from_records(records)
    out_df = out_df.set_index("date")
    out_df = out_df.sort_index()
    # ensure only YIELDS columns in order
    out_df = out_df.reindex(columns=YIELDS)
    out_df.to_csv(out_csv, index=True)
    print(f"Wrote {len(out_df)} rows and {len(out_df.columns)} columns to {out_csv}")


def main():
    repo_root = Path(__file__).resolve().parents[1]
    taux_dir = repo_root / "data" / "TAUX"
    out_csv = repo_root / "data" / "TAUX_combined.csv"
    merge_taux(taux_dir, out_csv)


if __name__ == "__main__":
    main()
