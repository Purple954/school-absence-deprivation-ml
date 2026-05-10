"""
clean.py - filter, join, and validate the raw data

Loads the four raw files from data/raw/, filters to state-funded
mainstream schools, removes suppressed values, joins school
characteristics from GIAS, and attaches IMD deprivation scores
via the postcode -> LSOA lookup.

Usage:
    python src/clean.py [--raw-dir data/raw] [--out-dir data/processed]

Outputs:
    data/processed/schools_clean.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# DfE establishment type codes that count as state-funded mainstream
# Source: DfE GIAS establishment type reference
STATE_MAINSTREAM_TYPES = {
    "Community school",
    "Voluntary aided school",
    "Voluntary controlled school",
    "Foundation school",
    "Academy sponsor led",
    "Academy converter",
    "Free school",
    "Studio school",
    "University technical college",
    "Foundation special school",
}

# DfE education phases to include (exclude nursery, 16+, special)
INCLUDE_PHASES = {"Primary", "Secondary", "All-through", "Middle deemed primary",
                  "Middle deemed secondary"}

# Suppression marker used in DfE absence files
SUPPRESSED = "z"


# Year keys matching ABSENCE_YEARS in ingest.py
_ABSENCE_YEAR_KEYS = ["1819", "2122", "2223", "2324"]


def _parse_absence_file(path: Path, year_key: str) -> pd.DataFrame:
    """Parse a single DfE school absence CSV and return a tidy DataFrame.

    Handles column name variations across DfE annual releases. Suppressed
    values are dropped. A ``year_key`` column is added.

    Parameters
    ----------
    path : Path
        Path to the absence CSV.
    year_key : str
        Short year code (e.g. "2324") added as a column.

    Returns
    -------
    pandas.DataFrame
        Columns: urn, [school_name, la_name, school_type, phase,]
        total_pupils, pa_rate, year_key.
    """
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    df.columns = (
        df.columns
        .str.replace(r"^[^\x00-\x7f]+", "", regex=True)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Some DfE CSVs have duplicate column names — drop extras before concat
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Older DfE files combine national/regional/LA/school rows — keep school only
    if "geographic_level" in df.columns:
        df = df[df["geographic_level"].str.strip().str.lower() == "school"].copy()

    # Some DfE files contain multiple years — keep only the most recent
    if "time_period" in df.columns:
        latest = df["time_period"].max()
        df = df[df["time_period"] == latest].copy()

    col_map = {}
    for col in df.columns:
        if col in ("urn", "school_urn"):
            col_map[col] = "urn"
        elif col == "school_name" or col == "school":
            col_map[col] = "school_name"
        elif "la_name" in col or "local_authority_name" in col:
            col_map[col] = "la_name"
        elif "school_type" in col or "establishment_type" in col or col == "academy_type":
            col_map[col] = "school_type"
        elif col in ("education_phase", "phase_of_education", "school_phase"):
            col_map[col] = "phase"
        elif col in ("enrolments",) or "number_of_pupils" in col or "headcount" in col or "total_pupils" in col:
            col_map[col] = "total_pupils"
        elif col == "enrolments_pa_10_exact_percent":
            col_map[col] = "pa_rate"
        elif "persistent_absence" in col and "percent" in col:
            col_map[col] = "pa_rate"
        elif col in ("pa_rate_percent", "pa_percent"):
            col_map[col] = "pa_rate"

    df = df.rename(columns=col_map)

    # Rename may create duplicates (two source cols mapping to same target) — deduplicate
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if "pa_rate" not in df.columns:
        for col in df.columns:
            if "pa_10" in col and "percent" in col:
                df = df.rename(columns={col: "pa_rate"})
                break
    if "pa_rate" not in df.columns:
        for col in df.columns:
            if "persistent" in col and ("rate" in col or "percent" in col or "pct" in col):
                df = df.rename(columns={col: "pa_rate"})
                break

    required = ["urn", "pa_rate"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Could not find columns {missing_cols} in {path.name}. "
            f"Available: {list(df.columns)}"
        )

    before = len(df)
    df = df[~df["pa_rate"].astype(str).str.strip().isin(["z", "x", "c", "."])]
    df["pa_rate"] = pd.to_numeric(df["pa_rate"], errors="coerce")
    df = df.dropna(subset=["pa_rate"])
    df["urn"] = pd.to_numeric(df["urn"], errors="coerce")
    df = df.dropna(subset=["urn"])
    df["urn"] = df["urn"].astype(int)
    df["year_key"] = year_key

    keep = [c for c in ["urn", "school_name", "la_name", "school_type", "phase",
                         "total_pupils", "pa_rate", "year_key"] if c in df.columns]
    print(f"  {year_key}: {len(df):,} schools ({before - len(df):,} suppressed rows dropped).")
    return df[keep].copy()


def load_absence_panel(raw_dir: Path) -> pd.DataFrame:
    """Load and stack all available DfE absence year files into a panel.

    Looks for files named ``absence_{year_key}_school.csv`` for each key in
    ``_ABSENCE_YEAR_KEYS``. Skips missing files silently.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the year CSV files.

    Returns
    -------
    pandas.DataFrame
        Stacked panel with a ``year_key`` column. One row per school-year.

    Raises
    ------
    FileNotFoundError
        If no absence files are found.
    """
    frames = []
    for year_key in _ABSENCE_YEAR_KEYS:
        path = raw_dir / f"absence_{year_key}_school.csv"
        if not path.exists():
            print(f"  [skip] absence_{year_key}_school.csv not found.")
            continue
        frames.append(_parse_absence_file(path, year_key))

    if not frames:
        raise FileNotFoundError(
            "No absence year files found in data/raw. Run src/ingest.py first."
        )

    panel = pd.concat(frames, ignore_index=True)
    years_loaded = panel["year_key"].unique().tolist()
    print(f"  Panel: {len(panel):,} school-year rows across years {years_loaded}.")
    return panel


def load_absence(raw_dir: Path) -> pd.DataFrame:
    """Load the 2023-24 DfE absence CSV (single-year, backward-compatible).

    For multi-year panel loading use :func:`load_absence_panel`.

    Parameters
    ----------
    raw_dir : Path
        Directory containing ``absence_2324_school.csv``.

    Returns
    -------
    pandas.DataFrame
        Columns: urn, school_name, la_name, school_type, phase,
        total_pupils, pa_rate, year_key.

    Raises
    ------
    FileNotFoundError
        If absence_2324_school.csv is missing.
    """
    path = raw_dir / "absence_2324_school.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run ingest.py first.")
    print("Loading absence data (single year 2023-24)...")
    return _parse_absence_file(path, "2324")


def load_gias(raw_dir: Path) -> pd.DataFrame:
    """Load the GIAS establishments extract and return relevant columns.

    Parameters
    ----------
    raw_dir : Path
        Directory containing gias_establishments.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: urn, ofsted_rating, urban_rural, region_name,
        number_of_pupils (from GIAS), postcode.

    Raises
    ------
    FileNotFoundError
        If gias_establishments.csv is missing.
    """
    path = raw_dir / "gias_establishments.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run ingest.py first.")

    print("Loading GIAS data...")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()

    # GIAS column name mapping - field names are stable across extracts
    rename = {
        "URN": "urn",
        "EstablishmentName": "school_name_gias",
        "EstablishmentTypeGroup (name)": "type_group",
        "TypeOfEstablishment (name)": "establishment_type",
        "PhaseOfEducation (name)": "phase_gias",
        "OfstedRating (name)": "ofsted_rating",
        "UrbanRural (name)": "urban_rural",
        "GOR (name)": "region_name",
        "NumberOfPupils": "number_of_pupils_gias",
        "PercentageFSM": "percent_fsm",
        "Postcode": "postcode",
        "StatutoryLowAge": "age_low",
        "StatutoryHighAge": "age_high",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "urn" not in df.columns:
        raise ValueError("URN column not found in GIAS file.")

    df["urn"] = pd.to_numeric(df["urn"], errors="coerce")
    df = df.dropna(subset=["urn"])
    df["urn"] = df["urn"].astype(int)

    keep = [c for c in ["urn", "establishment_type", "phase_gias", "ofsted_rating",
                         "urban_rural", "region_name", "number_of_pupils_gias",
                         "percent_fsm", "postcode", "type_group"] if c in df.columns]
    print(f"  {len(df):,} establishments in GIAS.")
    return df[keep].copy()


def load_imd(raw_dir: Path) -> pd.DataFrame:
    """Load IMD 2019 File 7 and return LSOA-level deprivation scores.

    Parameters
    ----------
    raw_dir : Path
        Directory containing imd_2019_scores.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: lsoa_code, imd_score, imd_decile.

    Raises
    ------
    FileNotFoundError
        If imd_2019_scores.csv is missing.
    """
    path = raw_dir / "imd_2019_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run ingest.py first.")

    print("Loading IMD 2019...")
    df = pd.read_csv(path, encoding="latin-1")

    rename = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc.startswith("lsoa code"):
            rename[col] = "lsoa_code"
        elif lc.startswith("index of multiple deprivation (imd) score"):
            rename[col] = "imd_score"
        elif lc.startswith("index of multiple deprivation (imd) decile"):
            rename[col] = "imd_decile"

    df = df.rename(columns=rename)
    print(f"  {len(df):,} LSOAs in IMD file.")
    return df[["lsoa_code", "imd_score", "imd_decile"]].copy()


def load_postcode_lookup(raw_dir: Path) -> pd.DataFrame:
    """Load the ONS postcode -> LSOA lookup.

    Parameters
    ----------
    raw_dir : Path
        Directory containing onspd_postcode_lsoa.csv.

    Returns
    -------
    pandas.DataFrame
        Columns: postcode (normalised, no spaces), lsoa_code.

    Raises
    ------
    FileNotFoundError
        If onspd_postcode_lsoa.csv is missing.
    """
    path = raw_dir / "onspd_postcode_lsoa.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}. Run ingest.py first.")

    print("Loading postcode lookup...")
    df = pd.read_csv(path, low_memory=False)

    # Identify postcode and LSOA columns by partial name match
    pc_col = next((c for c in df.columns if c.upper() in ("PCD", "PCD2", "PCDS", "POSTCODE")), None)
    lsoa_col = next((c for c in df.columns if "LSOA" in c.upper()), None)

    if pc_col is None or lsoa_col is None:
        raise ValueError(
            f"Could not identify postcode/LSOA columns in ONSPD file. "
            f"Columns found: {list(df.columns)}"
        )

    df = df[[pc_col, lsoa_col]].rename(columns={pc_col: "postcode_raw", lsoa_col: "lsoa_code"})
    df["postcode"] = df["postcode_raw"].str.upper().str.replace(" ", "", regex=False)
    df = df.dropna(subset=["postcode", "lsoa_code"])
    df = df.drop_duplicates("postcode")

    print(f"  {len(df):,} postcodes in lookup.")
    return df[["postcode", "lsoa_code"]].copy()


def filter_state_mainstream(df: pd.DataFrame) -> pd.DataFrame:
    """Filter the joined dataset to state-funded mainstream schools only.

    Uses establishment_type from GIAS if available, otherwise falls back
    to school_type from the absence file.

    Parameters
    ----------
    df : pandas.DataFrame
        Joined dataset with establishment_type and/or school_type columns.

    Returns
    -------
    pandas.DataFrame
        Filtered subset.
    """
    before = len(df)
    type_col = "establishment_type" if "establishment_type" in df.columns else "school_type"

    if type_col in df.columns:
        df = df[df[type_col].isin(STATE_MAINSTREAM_TYPES)].copy()

    phase_col = "phase_gias" if "phase_gias" in df.columns else "phase"
    if phase_col in df.columns:
        df = df[df[phase_col].isin(INCLUDE_PHASES)].copy()

    print(f"  Filtered to state mainstream: {len(df):,} schools (removed {before - len(df):,}).")
    return df


def build_clean_dataset(
    absence_df: pd.DataFrame,
    gias_df: pd.DataFrame,
    imd_df: pd.DataFrame,
    postcode_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join all four sources into a single clean dataset.

    Join order:
    1. absence_df LEFT JOIN gias_df on URN
    2. result LEFT JOIN postcode_df on normalised postcode
    3. result LEFT JOIN imd_df on LSOA code

    Parameters
    ----------
    absence_df : pandas.DataFrame
        From load_absence().
    gias_df : pandas.DataFrame
        From load_gias().
    imd_df : pandas.DataFrame
        From load_imd().
    postcode_df : pandas.DataFrame
        From load_postcode_lookup().

    Returns
    -------
    pandas.DataFrame
        One row per school with absence rate, school characteristics,
        and deprivation score.
    """
    print("Building clean dataset...")

    # Join GIAS
    df = absence_df.merge(gias_df, on="urn", how="left")

    # Normalise postcode for join
    if "postcode" in df.columns:
        df["postcode_key"] = df["postcode"].str.upper().str.replace(" ", "", regex=False)
        postcode_df = postcode_df.rename(columns={"postcode": "postcode_key"})
        df = df.merge(postcode_df, on="postcode_key", how="left")
        df = df.drop(columns="postcode_key")

    # Join IMD via LSOA
    if "lsoa_code" in df.columns:
        df = df.merge(imd_df, on="lsoa_code", how="left")

    # Filter to state mainstream
    df = filter_state_mainstream(df)

    # Composite panel ID: unique per school-year (or just URN for single-year)
    if "year_key" in df.columns:
        df["panel_id"] = df["urn"].astype(str) + "_" + df["year_key"].astype(str)
    else:
        df["panel_id"] = df["urn"].astype(str)

    # Report join completeness
    for col, label in [("ofsted_rating", "Ofsted rating"),
                        ("imd_score", "IMD score"),
                        ("lsoa_code", "LSOA code")]:
        if col in df.columns:
            missing = df[col].isna().sum()
            pct = missing / len(df) * 100
            print(f"  {label}: {len(df) - missing:,}/{len(df):,} matched ({pct:.1f}% missing)")

    return df.reset_index(drop=True)


def main(raw_dir: str = "data/raw", out_dir: str = "data/processed") -> pd.DataFrame:
    """Run the full cleaning pipeline.

    Parameters
    ----------
    raw_dir : str, optional
        Path to raw data directory.
    out_dir : str, optional
        Path for processed outputs.

    Returns
    -------
    pandas.DataFrame
        The clean joined dataset.
    """
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Prefer multi-year panel; fall back to single 2023-24 file
    try:
        absence_df = load_absence_panel(raw)
        print(f"  Panel mode: {absence_df['year_key'].nunique()} year(s) loaded.")
    except FileNotFoundError:
        absence_df = load_absence(raw)

    gias_df = load_gias(raw)
    imd_df = load_imd(raw)
    postcode_df = load_postcode_lookup(raw)

    clean = build_clean_dataset(absence_df, gias_df, imd_df, postcode_df)

    out_path = out / "schools_clean.csv"
    clean.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({clean.shape[0]:,} rows x {clean.shape[1]} columns)")
    return clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean and join data for school absence ML project."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    main(raw_dir=args.raw_dir, out_dir=args.out_dir)
