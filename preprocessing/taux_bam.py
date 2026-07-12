from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "TAUX" / "raw" / "Data.xlsm"
DAILY_OUTPUT_PATH = ROOT / "data" / "TAUX" / "processed" / "taux_bam.csv"
MONTHLY_OUTPUT_PATH = ROOT / "data" / "TAUX" / "processed" / "taux_bam_monthly.csv"

YIELDS = [
    "3M", "6M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y",
    "10Y", "11Y", "12Y", "13Y", "14Y", "15Y", "16Y", "17Y", "18Y", "19Y",
    "20Y", "30Y"
]

RAW_COLUMN_MAP = {
    "Date": "Date",
    "13 sem": "3M",
    "26 sem": "6M",
    "52 sem": "1Y",
    "2 ans": "2Y",
    "5 ans": "5Y",
    "10 ans": "10Y",
    "15 ans": "15Y",
    "20 ans": "20Y",
    "30 ans": "30Y",
}

MATURITY_BY_TENOR = {
    "3M": 0.25,
    "6M": 0.5,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "4Y": 4.0,
    "5Y": 5.0,
    "6Y": 6.0,
    "7Y": 7.0,
    "8Y": 8.0,
    "9Y": 9.0,
    "10Y": 10.0,
    "11Y": 11.0,
    "12Y": 12.0,
    "13Y": 13.0,
    "14Y": 14.0,
    "15Y": 15.0,
    "16Y": 16.0,
    "17Y": 17.0,
    "18Y": 18.0,
    "19Y": 19.0,
    "20Y": 20.0,
    "30Y": 30.0,
}


def to_monthly_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily curves to monthly mean per tenor."""
    monthly = df.resample("ME").mean()
    monthly.index = monthly.index + pd.offsets.MonthBegin(1)
    return monthly


def build_standardized_curve(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Rename existing columns and interpolate missing tenors linearly by maturity."""
    df = raw_df.copy()
    df = df.rename(columns={col: RAW_COLUMN_MAP[col] for col in RAW_COLUMN_MAP if col in df.columns})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    standardized = pd.DataFrame({"Date": df["Date"]})
    for tenor in YIELDS:
        if tenor in df.columns:
            standardized[tenor] = pd.to_numeric(df[tenor], errors="coerce")
        else:
            standardized[tenor] = np.nan

    for idx in range(len(standardized)):
        available = {
            tenor: float(standardized.loc[idx, tenor])
            for tenor in YIELDS
            if pd.notna(standardized.loc[idx, tenor])
        }
        if len(available) < 2:
            continue

        ordered = sorted(available.items(), key=lambda item: MATURITY_BY_TENOR[item[0]])
        maturity_points = [MATURITY_BY_TENOR[tenor] for tenor, _ in ordered]
        yield_points = [value for _, value in ordered]

        for tenor in YIELDS:
            if pd.notna(standardized.loc[idx, tenor]):
                continue
            maturity = MATURITY_BY_TENOR[tenor]
            if maturity <= maturity_points[0]:
                standardized.loc[idx, tenor] = yield_points[0]
            elif maturity >= maturity_points[-1]:
                standardized.loc[idx, tenor] = yield_points[-1]
            else:
                standardized.loc[idx, tenor] = np.interp(maturity, maturity_points, yield_points)

    standardized = standardized.set_index("Date")
    standardized = standardized.astype(float)
    return standardized


def main() -> None:
    raw_df = pd.read_excel(RAW_PATH)
    standardized = build_standardized_curve(raw_df)

    standardized.to_csv(DAILY_OUTPUT_PATH, index=True)

    monthly = to_monthly_mean(standardized)
    monthly.to_csv(MONTHLY_OUTPUT_PATH, index=True)

    print(standardized.head())
    print("\nMonthly sample:")
    print(monthly.head())


if __name__ == "__main__":
    main()