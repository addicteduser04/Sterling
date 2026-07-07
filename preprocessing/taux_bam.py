import re
import pandas as pd  # type: ignore
from pathlib import Path

YIELDS = ['3M','6M','1Y','2Y','3Y','4Y','5Y','6Y','7Y','8Y','9Y','10Y',
          '11Y','12Y','13Y','14Y','15Y','16Y','17Y','18Y','19Y','20Y','30Y']

RAW_YIELDS = {
    '13 semaines': '3M', '26 semaines': '6M', '52 semaines': '1Y',
    '2 ans': '2Y', '5 ans': '5Y', '10 ans': '10Y',
    '15 ans': '15Y', '20 ans': '20Y', '30 ans': '30Y',
}
# Normalized lookup, tolerant of truncated/whitespace variants like "13 semain"
RAW_YIELDS_NORM = {k.strip().lower(): v for k, v in RAW_YIELDS.items()}

FORMAT_SWITCH_DATE = "20250903"  # new (filename-dated) format starts here
FOLDER = Path("../data/TAUX/raw")
FILENAME_DATE_RE = re.compile(r"(\d{8})")


def linear_interpolation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing tenor points cross-sectionally, per date."""
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.sort_index()
    df = df.interpolate(method='linear', axis=1, limit_direction='both')
    return df


def match_tenor(label: str) -> str | None:
    """Map a raw tenor label (possibly truncated) to a standardized code."""
    norm = str(label).strip().lower()
    if norm in RAW_YIELDS_NORM:
        return RAW_YIELDS_NORM[norm]
    for raw_key, code in RAW_YIELDS_NORM.items():
        if raw_key.startswith(norm) or norm.startswith(raw_key):
            return code
    return None


def extract_date_from_filename(name: str) -> pd.Timestamp:
    m = FILENAME_DATE_RE.search(name)
    if not m:
        raise ValueError(f"no YYYYMMDD date found in filename: {name}")
    return pd.to_datetime(m.group(1), format='%Y%m%d')


def load_old_format(path: Path) -> pd.DataFrame:
    """Pre-20250903: wide grid, dates as columns (before transpose), 23 tenors."""
    raw = pd.read_excel(path)
    raw = raw.T
    raw = raw.drop("Tenor", errors="ignore")

    parsed_index = pd.to_datetime(raw.index, format='%d/%m/%Y', errors='coerce', dayfirst=True)
    bad_rows = parsed_index.isna()
    if bad_rows.any():
        print(f"  [warn] {path.name}: dropping non-date rows {list(raw.index[bad_rows])}")
        raw = raw.loc[~bad_rows]
        parsed_index = parsed_index[~bad_rows]
    raw.index = parsed_index

    if raw.shape[1] != len(YIELDS):
        raise ValueError(f"expected {len(YIELDS)} cols, got {raw.shape[1]}")
    raw.columns = YIELDS
    return raw


def load_new_format(path: Path) -> pd.DataFrame:
    """From 20250903: single snapshot per file, Tenor + value column, date in filename."""
    raw = pd.read_excel(path)
    if raw.shape[1] < 2:
        raise ValueError(f"expected at least 2 columns, got {raw.shape[1]}")

    tenor_col, value_col = raw.columns[0], raw.columns[1]
    raw = raw[[tenor_col, value_col]].copy()

    codes = raw[tenor_col].map(match_tenor)
    unmapped = raw.loc[codes.isna(), tenor_col].tolist()
    if unmapped:
        print(f"  [warn] {path.name}: unmapped tenor labels skipped: {unmapped}")

    raw = raw.loc[codes.notna()]
    codes = codes.loc[codes.notna()]

    date = extract_date_from_filename(path.name)
    row = pd.Series(raw[value_col].values, index=codes.values, name=date)
    row = row.groupby(level=0).last()  # guard against duplicate tenor rows

    df_row = row.to_frame().T
    df_row = df_row.reindex(columns=YIELDS)
    return df_row


def load_curve_file(path: Path, folder_name: str) -> pd.DataFrame:
    if folder_name < FORMAT_SWITCH_DATE:
        return load_old_format(path)
    return load_new_format(path)


def load_all_curves(folder: Path) -> pd.DataFrame:
    frames = []
    for item in sorted(folder.iterdir()):
        if not item.is_dir():
            continue
        excel_dir = item / "excel"
        if not excel_dir.is_dir():
            continue
        for curve_file in sorted(excel_dir.iterdir()):
            try:
                frames.append(load_curve_file(curve_file, item.name))
            except Exception as e:
                print(f"  [error] failed on {item.name}/excel/{curve_file.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=YIELDS)

    data = pd.concat(frames, axis=0)
    data = data[~data.index.duplicated(keep='last')]
    data = data.sort_index()
    return data

def to_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily curves to monthly mean per tenor."""
    monthly = df.resample('ME').mean()
    monthly.index = (monthly.index + pd.offsets.MonthBegin(1)) 
    return monthly

data = load_all_curves(FOLDER)
data_interp = linear_interpolation(data)
data_monthly = to_monthly_mean(data_interp)
data_interp.to_csv("../data/TAUX/processed/taux_bam.csv", index_label="Date")
data_monthly.to_csv("../data/TAUX/processed/taux_bam_monthly.csv", index_label="Date")
print(data_interp)
