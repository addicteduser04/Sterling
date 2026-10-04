import pandas as pd
import pytest

from src.modelling.macro_data import asof_macro_features, matched_complete_origins, validate_observations
from src.modelling.phase3_data import parse_cpi, parse_interbank_exports, parse_policy_rate
from src.modelling.phase3_weekly_extract import extract_structured_weekly


def observations():
    return pd.DataFrame({
        "series_id": ["cpi", "cpi", "policy"],
        "reference_date": ["2025-05-31", "2025-06-30", "2025-06-15"],
        "publication_date": ["2025-06-20", "2025-07-20", "2025-06-15"],
        "available_from": ["2025-06-20", "2025-07-20", "2025-06-15"],
        "value": [100.0, 101.0, .025], "vintage": ["first", "first", "event"],
        "source": ["HCP", "HCP", "BAM"], "frequency": ["monthly", "monthly", "event"],
        "unit": ["index", "index", "decimal_rate"], "is_revised": [False] * 3,
        "timing_quality": ["VERIFIED_DATE"] * 3,
        "vintage_quality": ["FIRST_RELEASE", "FIRST_RELEASE", "UNREVISED_ADMIN"],
        "metadata": ["", "", ""],
    })


def test_cpi_reference_month_is_inaccessible_before_publication():
    result = asof_macro_features(pd.to_datetime(["2025-07-10", "2025-07-21"]), observations(), ["cpi"])
    assert result.iloc[0, 0] == 100.0
    assert result.iloc[1, 0] == 101.0


def test_no_backward_fill_and_exact_release_is_available():
    result = asof_macro_features(
        pd.to_datetime(["2025-06-01", "2025-06-20"]), observations(), ["cpi"]
    )
    assert pd.isna(result.iloc[0, 0])
    assert result.iloc[1, 0] == 100.0


def test_stale_limit_invalidates_but_does_not_replace():
    result = asof_macro_features(
        pd.to_datetime(["2025-06-16", "2025-07-20"]), observations(), ["policy"],
        max_stale_days={"policy": 30},
    )
    assert result.iloc[0, 0] == .025
    assert pd.isna(result.iloc[1, 0])


def test_availability_cannot_precede_publication():
    bad = observations()
    bad.loc[0, "available_from"] = "2025-06-19"
    with pytest.raises(ValueError, match="cannot precede"):
        validate_observations(bad)


def test_common_period_requires_same_complete_origins():
    idx = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"])
    curve = pd.DataFrame({"curve": [1, 2, 3]}, index=idx)
    macro = pd.DataFrame({"macro": [None, 4, 5]}, index=idx)
    assert list(matched_complete_origins(curve, macro)) == list(idx[1:])


def test_policy_parser_uses_decimals_and_decision_date(tmp_path):
    path = tmp_path / "policy.csv"
    path.write_text("Date,Taux directeur\n20/03/2025,2.25%\n", encoding="utf-8")
    parsed = parse_policy_rate(path)
    assert parsed.value.iloc[0] == .0225
    assert parsed.available_from.iloc[0] == pd.Timestamp("2025-03-20")


def test_cpi_parser_uses_release_not_reference_date(tmp_path):
    values = tmp_path / "cpi.csv"
    values.write_text("Mois,IPC\n2025M5,120.4\n", encoding="utf-8")
    calendar = tmp_path / "calendar.csv"
    calendar.write_text(
        "reference_period,publication_date,source_url,status\n"
        "2025-05,2025-06-20,https://www.hcp.ma/release,verified_first_release\n",
        encoding="utf-8",
    )
    parsed = parse_cpi(values, calendar)
    assert parsed.reference_date.iloc[0] == pd.Timestamp("2025-05-31")
    assert parsed.available_from.iloc[0] == pd.Timestamp("2025-06-20")


def test_cpi_parser_rejects_partial_release_calendar(tmp_path):
    values = tmp_path / "cpi.csv"
    values.write_text("Mois,IPC\n2025M4,121.5\n2025M5,120.4\n", encoding="utf-8")
    calendar = tmp_path / "calendar.csv"
    calendar.write_text(
        "reference_period,publication_date,source_url,status\n"
        "2025-05,2025-06-20,https://www.hcp.ma/release,verified_first_release\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="calendar incomplete"):
        parse_cpi(values, calendar)


def test_interbank_parser_uses_next_origin_and_keeps_gap(tmp_path):
    oldest = tmp_path / "oldest.csv"
    newest = tmp_path / "newest.csv"
    header = '"Marché monétaire interbancaire"\n"En millions de dirhams"\nDate;"Taux Moyen Pondéré";"Volume JJ";Encours\n'
    oldest.write_text(header + '01/01/2001;"6,315 %";0;2917\n', encoding="utf-8")
    newest.write_text(header + '10/11/2023;"3,000 %";1963;6843\n', encoding="utf-8")
    parsed = parse_interbank_exports([oldest, newest])
    assert list(parsed.reference_date.dt.year) == [2001, 2023]
    assert parsed.value.iloc[0] == pytest.approx(.06315)
    assert parsed.available_from.iloc[0] == pd.Timestamp("2001-01-02")
    assert set(parsed.timing_quality) == {"CONSERVATIVE_NEXT_ORIGIN"}


def test_weekly_extractor_never_invents_publication_or_demanded_amount():
    text = """
    Semaine Semaine du 10-11-22 du 17-11-22 au 16-11-22 au 23-11-22
    MARCHE MONETAIRE Interventions* de Bank Al-Maghrib
    Interventions de BAM        104 802        90 132
    -Avances à 7 jours          46 790         32 120
    Résultats des avances à 7 jours sur appel d'offres du 23/11/2022
    Montant servi 50 900
    MARCHE DES ADJUDICATIONS
    Adjudications au 22-11-22
    13 semaines        1 360        1 700        2,58
    INFLATION
    """
    liquidity, auctions = extract_structured_weekly(text, "official.pdf")
    assert {row["series_id"] for row in liquidity} == {
        "bam_liquidity_total_interventions", "bam_liquidity_7d_advances_stock",
        "bam_liquidity_7d_allotment",
    }
    assert all(row["timing_quality"] == "REFERENCE_DATE_ONLY" for row in liquidity)
    assert auctions[0]["result_publication_date"] is None
    assert auctions[0]["amount_demanded_mad_millions"] is None
    assert auctions[0]["amount_accepted_mad_millions"] == 1700
