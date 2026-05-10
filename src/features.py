"""
features.py - feature engineering for the school absence ML pipeline

Transforms the clean joined dataset into a model-ready feature matrix.
Handles encoding, scaling flags, and missing value imputation.

Usage:
    python src/features.py [--data data/processed/schools_clean.csv]
                           [--out-dir data/processed]

Outputs:
    data/processed/features.csv   - feature matrix (X) with school URN
    data/processed/target.csv     - target vector (y) with school URN
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Ofsted rating string -> numeric (lower = better, consistent with DfE convention)
OFSTED_MAP = {
    "Outstanding": 1,
    "Good": 2,
    "Requires improvement": 3,
    "Inadequate": 4,
    "Serious Weaknesses": 4,
    "Special Measures": 4,
}

# Urban/rural classification -> binary urban flag
# ONS 2011 urban/rural codes: A1/B1 = urban >10k, C1/D1 = town/fringe, E1/F1 = rural
URBAN_KEYWORDS = ["urban", "city", "town"]

# Phase -> numeric encoding
PHASE_MAP = {
    "Primary": 1,
    "Middle deemed primary": 1,
    "Secondary": 2,
    "Middle deemed secondary": 2,
    "All-through": 3,
}

# Regions in England (used for dummy encoding)
REGIONS = [
    "North East", "North West", "Yorkshire and The Humber",
    "East Midlands", "West Midlands", "East of England",
    "London", "South East", "South West",
]

# Year key -> numeric value for use as a continuous/ordinal feature.
# COVID years (2019-20, 2020-21) are excluded so gaps reflect real time.
YEAR_NUMERIC = {
    "1819": 0,
    "2122": 3,
    "2223": 4,
    "2324": 5,
}


def encode_ofsted(series: pd.Series) -> pd.Series:
    """Map Ofsted rating strings to numeric scores 1-4.

    Parameters
    ----------
    series : pandas.Series
        Raw Ofsted rating strings from GIAS.

    Returns
    -------
    pandas.Series
        Integer scores 1 (Outstanding) to 4 (Inadequate). Schools with
        no Ofsted rating get NaN.
    """
    return series.map(OFSTED_MAP)


def encode_urban_rural(series: pd.Series) -> pd.Series:
    """Convert urban/rural classification to a binary urban flag.

    Parameters
    ----------
    series : pandas.Series
        Urban/rural category strings from GIAS.

    Returns
    -------
    pandas.Series
        1 if urban, 0 if rural/town, NaN if unknown.
    """
    lower = series.str.lower().fillna("")
    return lower.apply(
        lambda x: 1 if any(k in x for k in URBAN_KEYWORDS) else (0 if x else np.nan)
    )


def log_school_size(series: pd.Series) -> pd.Series:
    """Log-transform school size to reduce right skew.

    Parameters
    ----------
    series : pandas.Series
        Number of pupils (raw count).

    Returns
    -------
    pandas.Series
        log(1 + n_pupils).
    """
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log1p(numeric)


def make_deprivation_quintile(imd_decile: pd.Series) -> pd.Series:
    """Derive IMD quintile from IMD decile.

    Parameters
    ----------
    imd_decile : pandas.Series
        IMD decile values 1-10 (1 = most deprived).

    Returns
    -------
    pandas.Series
        Quintile values 1-5 (1 = most deprived).
    """
    return np.ceil(imd_decile / 2).astype("Int64")


def encode_phase(series: pd.Series) -> pd.Series:
    """Map school phase to a numeric label.

    Parameters
    ----------
    series : pandas.Series
        Phase of education strings.

    Returns
    -------
    pandas.Series
        1 = Primary, 2 = Secondary, 3 = All-through. NaN for unknowns.
    """
    return series.map(PHASE_MAP)


def make_region_dummies(series: pd.Series) -> pd.DataFrame:
    """One-hot encode the region column, dropping the first category.

    Parameters
    ----------
    series : pandas.Series
        Region name strings.

    Returns
    -------
    pandas.DataFrame
        Binary dummy columns, one per region (minus the reference).
    """
    dummies = pd.get_dummies(series, prefix="region", drop_first=True)
    return dummies.astype(int)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the model-ready feature matrix from the clean dataset.

    Features produced:
    - year_numeric       : numeric year (panel only; 0=2018-19 ... 5=2023-24)
    - imd_score          : continuous deprivation score (higher = more deprived)
    - imd_quintile       : categorical deprivation quintile 1-5
    - ofsted_numeric     : Ofsted rating 1 (Outstanding) to 4 (Inadequate)
    - log_pupils         : log(1 + number of pupils)
    - phase_numeric      : 1=Primary, 2=Secondary, 3=All-through
    - is_urban           : 1 if urban school location, 0 if rural
    - region_*           : one-hot encoded region dummies (reference = East Midlands)

    The index is ``panel_id`` ({urn}_{year_key}) when panel data is present,
    otherwise ``urn``.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of clean.build_clean_dataset(), loaded from schools_clean.csv.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix indexed by panel_id (or urn for single-year).
    y : pandas.Series
        Target variable (persistent absence rate %) with same index.
    """
    print("Building feature matrix...")

    df = df.copy()

    is_panel = "panel_id" in df.columns and "year_key" in df.columns
    if is_panel:
        df = df.set_index("panel_id")
    elif "urn" in df.columns:
        df = df.set_index("urn")

    # Target
    y = pd.to_numeric(df["pa_rate"], errors="coerce")

    # Continuous deprivation
    X = pd.DataFrame(index=df.index)

    # Year as numeric feature (panel only)
    if is_panel and "year_key" in df.columns:
        X["year_numeric"] = df["year_key"].astype(str).map(YEAR_NUMERIC)

    X["imd_score"] = pd.to_numeric(df.get("imd_score"), errors="coerce")
    X["imd_quintile"] = make_deprivation_quintile(
        pd.to_numeric(df.get("imd_decile"), errors="coerce")
    )

    # School characteristics
    ofsted_col = "ofsted_rating" if "ofsted_rating" in df.columns else None
    if ofsted_col:
        X["ofsted_numeric"] = encode_ofsted(df[ofsted_col])

    # Free School Meals % — strong proxy for school-level deprivation
    if "percent_fsm" in df.columns:
        X["percent_fsm"] = pd.to_numeric(df["percent_fsm"], errors="coerce")

    # School size - prefer GIAS count, fall back to absence file count
    size_col = next(
        (c for c in ["number_of_pupils_gias", "total_pupils", "number_of_pupils"] if c in df.columns),
        None,
    )
    X["log_pupils"] = log_school_size(df[size_col]) if size_col else np.nan

    # Phase
    phase_col = "phase_gias" if "phase_gias" in df.columns else "phase"
    X["phase_numeric"] = encode_phase(df.get(phase_col, pd.Series(dtype=str)))

    # Urban/rural
    X["is_urban"] = encode_urban_rural(df.get("urban_rural", pd.Series(dtype=str)))

    # Region dummies
    if "region_name" in df.columns:
        region_dummies = make_region_dummies(df["region_name"])
        X = pd.concat([X, region_dummies], axis=1)

    # Drop rows where target is missing
    valid = y.notna()
    X = X[valid]
    y = y[valid]

    # Drop columns that are entirely NaN (e.g. optional GIAS fields absent from extract)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        print(f"  Dropping all-NaN columns (not in this GIAS extract): {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)

    # Impute remaining NaNs with column medians (simple, interpretable)
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)

    X = X.astype(float)

    print(f"  Feature matrix: {X.shape[0]:,} schools x {X.shape[1]} features")
    print(f"  Features: {list(X.columns)}")
    missing_target = (~valid).sum()
    if missing_target:
        print(f"  Dropped {missing_target:,} schools with missing PA rate.")

    return X, y


def main(data_path: str = "data/processed/schools_clean.csv",
         out_dir: str = "data/processed") -> None:
    """Run feature engineering and save outputs.

    Parameters
    ----------
    data_path : str
        Path to the clean dataset from clean.py.
    out_dir : str
        Directory to write feature matrix and target vector.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} rows from {data_path}")

    X, y = build_feature_matrix(df)

    X.to_csv(out / "features.csv")
    y.to_csv(out / "target.csv", header=True)
    print(f"\nSaved features.csv and target.csv to {out}")

    # Save grouping metadata for model.py (school-level split + spatial CV)
    is_panel = "panel_id" in df.columns and "year_key" in df.columns
    if is_panel:
        groups = df[["panel_id", "urn", "year_key"]].copy()
        if "region_name" in df.columns:
            groups["region_name"] = df["region_name"].values
        groups = groups.set_index("panel_id")
    else:
        groups = df[["urn"]].copy()
        if "region_name" in df.columns:
            groups["region_name"] = df["region_name"].values
        groups = groups.set_index("urn") if "urn" in df.columns else groups

    # Align groups to the feature matrix index (rows with valid target)
    groups = groups.loc[groups.index.isin(X.index)]
    groups.to_csv(out / "groups.csv")
    print(f"Saved groups.csv ({len(groups):,} rows) to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering for school absence ML project."
    )
    parser.add_argument("--data", default="data/processed/schools_clean.csv")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    main(data_path=args.data, out_dir=args.out_dir)
