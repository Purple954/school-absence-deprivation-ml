"""
ingest.py - download raw data for the school absence deprivation ML project

Downloads or guides the download of four open government datasets:
  1. DfE persistent absence statistics (2023-24) - requires one manual step
  2. Get Information About Schools (GIAS) bulk extract - auto
  3. Index of Multiple Deprivation 2019 - auto
  4. ONS Postcode Directory (ONSPD) - for postcode -> LSOA lookup - auto

Usage:
    python src/ingest.py [--data-dir data/raw]
"""

import argparse
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Academic years available from DfE Explore Education Statistics
# Keys are short codes used in filenames; values are URL slugs.
# 2019-20 and 2020-21 are excluded — DfE suppressed/didn't publish
# school-level data due to COVID school closures.
# ---------------------------------------------------------------------------

ABSENCE_YEARS = {
    "1819": "2018-19",
    "2122": "2021-22",
    "2223": "2022-23",
    "2324": "2023-24",
}

# ---------------------------------------------------------------------------
# Direct download URLs
# ---------------------------------------------------------------------------

# GIAS open data - all establishments extract (UTF-8 CSV)
# Source: get-information-schools.service.gov.uk/Downloads
_GIAS_URL = (
    "https://ea-edubase-api-prod.azurewebsites.net/edubase/downloads/public/"
    "edubasealldata%s.csv"
)

# IMD 2019 File 7 - scores, ranks, deciles for all English LSOAs
_IMD_URL = (
    "https://assets.publishing.service.gov.uk/government/uploads/system/uploads/"
    "attachment_data/file/845345/"
    "File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv"
)

# ONS Postcode Directory - November 2023 edition (lightweight subset)
# This endpoint returns a zip; we extract just the CSV we need
_ONSPD_URL = (
    "https://www.arcgis.com/sharing/rest/content/items/"
    "dc23a64fa2e34c6b98edef7887e8f0a3/data"
)


def _download_file(url: str, dest: Path, desc: str, timeout: int = 120) -> None:
    """Stream a file from url to dest with a progress bar.

    Parameters
    ----------
    url : str
        Direct download URL.
    dest : Path
        Destination file path.
    desc : str
        Label shown in the progress bar.
    timeout : int, optional
        Request timeout in seconds.
    """
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in r.iter_content(chunk_size=65536):
            fh.write(chunk)
            bar.update(len(chunk))


def download_absence_year(raw_dir: Path, year_key: str) -> Path:
    """Check for one year's DfE school absence CSV and print download instructions if missing.

    Parameters
    ----------
    raw_dir : Path
        Directory where the file should be saved.
    year_key : str
        Short year code from ABSENCE_YEARS (e.g. "2324").

    Returns
    -------
    Path
        Expected file path (may not yet exist if not downloaded).
    """
    year_label = ABSENCE_YEARS[year_key]
    dest = raw_dir / f"absence_{year_key}_school.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    print(f"""
  *** MANUAL DOWNLOAD REQUIRED: {year_label} ***

  1. Go to:
     https://explore-education-statistics.service.gov.uk/find-statistics/pupil-absence-in-schools-in-england/{year_label}

  2. Click "Download associated files" -> "Download all data (zip)"
     OR scroll to "Explore data and files" -> "Download files" and tick the
     school-level absence file (e.g. "1a_absence_3term_school.csv").

  3. Rename the file to:
       absence_{year_key}_school.csv

  4. Move it to:
       {dest.resolve()}

  Then re-run this script.
    """)
    return dest


def download_all_absence_years(raw_dir: Path) -> list:
    """Prompt download of all configured absence year files.

    Parameters
    ----------
    raw_dir : Path
        Directory where files should be saved.

    Returns
    -------
    list of Path
        Expected paths for each year (files may not yet exist).
    """
    paths = []
    for key in ABSENCE_YEARS:
        paths.append(download_absence_year(raw_dir, key))
    return paths


def download_absence_data(raw_dir: Path) -> Path:
    """Check for the DfE school absence statistics CSV and print instructions if missing.

    The DfE publishes school-level absence data through the Explore Education
    Statistics (EES) platform, which does not expose a direct programmatic
    download URL. This function checks whether the file has already been
    downloaded manually. If not, it prints step-by-step instructions.

    Parameters
    ----------
    raw_dir : Path
        Directory where the absence CSV should be saved.

    Returns
    -------
    Path
        Path where the file should be saved (may not yet exist).
    """
    dest = raw_dir / "absence_2324_school.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    print("""
  *** MANUAL DOWNLOAD REQUIRED (one time only) ***

  The DfE school-level absence file requires a manual download:

  1. Go to:
     https://explore-education-statistics.service.gov.uk/find-statistics/pupil-absence-in-schools-in-england/2023-24

  2. Click "Download associated files" -> "Download all data (zip)"
     OR scroll to "Explore data and files" -> "Download files" and tick:
       "Absence by school (1a_absence_3term_school.csv)"

  3. Save / extract the file and rename it to:
       absence_2324_school.csv

  4. Move it to:
       {dest}

  Then re-run this script.
  *************************************************
    """.format(dest=dest.resolve()))

    return dest


def download_gias(raw_dir: Path) -> Path:
    """Download the GIAS all-establishments bulk extract.

    GIAS (Get Information About Schools) provides a daily-updated CSV of
    all registered schools and their characteristics. The file is named
    with the current date in YYYYMMDD format; this function tries today's
    date and falls back to a recent known-good date.

    Parameters
    ----------
    raw_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to the saved GIAS CSV.
    """
    from datetime import date

    dest = raw_dir / "gias_establishments.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    print("  Downloading GIAS establishments extract...")

    # Try the last 30 days - GIAS updates periodically on weekdays
    for offset in range(30):
        from datetime import timedelta
        d = date.today() - timedelta(days=offset)
        url = _GIAS_URL % d.strftime("%Y%m%d")
        try:
            # HEAD is not supported on this server; use a streaming GET probe instead
            r = requests.get(url, timeout=15, stream=True)
            if r.status_code == 200:
                r.close()
                _download_file(url, dest, desc="  GIAS CSV")
                print(f"  Saved to {dest.name}")
                return dest
            r.close()
        except requests.RequestException:
            continue

    raise RuntimeError(
        "Could not download GIAS data. Visit https://get-information-schools.service.gov.uk/Downloads "
        "and save the 'All establishments' CSV as data/raw/gias_establishments.csv"
    )


def download_imd(raw_dir: Path) -> Path:
    """Download the Index of Multiple Deprivation 2019 File 7 CSV.

    Contains IMD scores, ranks, and deciles for all English LSOAs.

    Parameters
    ----------
    raw_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to the saved CSV.
    """
    dest = raw_dir / "imd_2019_scores.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    print("  Downloading IMD 2019...")
    _download_file(_IMD_URL, dest, desc="  IMD CSV")
    print(f"  Saved to {dest.name}")
    return dest


def download_onspd(raw_dir: Path) -> Path:
    """Build a postcode -> LSOA lookup for the school postcodes in GIAS.

    Rather than downloading the full ONS Postcode Directory (2.7M rows),
    this reads the school postcodes from gias_establishments.csv and
    queries the ONS Postcode Best-Fit FeatureServer for only those postcodes.
    Result is a compact lookup with ~20k rows covering all English schools.

    Parameters
    ----------
    raw_dir : Path
        Destination directory. Must already contain gias_establishments.csv.

    Returns
    -------
    Path
        Path to the saved postcode lookup CSV.
    """
    dest = raw_dir / "onspd_postcode_lsoa.csv"
    if dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    import pandas as pd

    gias_path = raw_dir / "gias_establishments.csv"
    if not gias_path.exists():
        raise FileNotFoundError(
            "gias_establishments.csv must be downloaded before onspd. "
            "Run download_gias() first."
        )

    gias = pd.read_csv(gias_path, encoding="latin-1", low_memory=False)
    gias.columns = gias.columns.str.strip()
    pc_col = next((c for c in gias.columns if c == "Postcode"), None)
    if pc_col is None:
        pc_col = next((c for c in gias.columns if "postcode" in c.lower()), None)

    postcodes = (
        gias[pc_col].dropna()
        .str.upper()
        .str.replace(" ", "", regex=False)
        .unique()
        .tolist()
    ) if pc_col else []

    print(f"  Found {len(postcodes):,} unique school postcodes in GIAS.")
    if not postcodes:
        raise ValueError("No postcodes found in GIAS file.")

    _download_ons_postcode_service(raw_dir, dest, postcodes=postcodes)
    return dest


def _download_ons_postcode_service(
    raw_dir: Path, dest: Path, postcodes: list = None
) -> None:
    """Look up LSOA codes for school postcodes via the postcodes.io bulk API.

    Queries postcodes.io in batches of 100 postcodes. Returns 2011 LSOA codes
    (lsoa11cd) which match the IMD 2019 geography. postcodes.io has full
    coverage including large-user and recently allocated postcodes that the
    ONS FeatureServer omits.

    Parameters
    ----------
    raw_dir : Path
        Raw data directory (unused but kept for interface consistency).
    dest : Path
        Destination CSV path.
    postcodes : list of str, optional
        Normalised (no spaces, uppercase) postcodes to look up.
    """
    import pandas as pd

    api_url = "https://api.postcodes.io/postcodes"

    def _add_space(pc: str) -> str:
        pc = pc.strip().upper().replace(" ", "")
        return pc[:-3] + " " + pc[-3:] if len(pc) >= 5 else pc

    if not postcodes:
        raise ValueError("postcodes list is required for postcodes.io lookup.")

    spaced = [_add_space(pc) for pc in postcodes]
    batch_size = 100
    batches = [spaced[i:i + batch_size] for i in range(0, len(spaced), batch_size)]
    print(f"  Fetching LSOA codes for {len(spaced):,} postcodes ({len(batches)} batches via postcodes.io)...")

    rows = []
    for batch in tqdm(batches, desc="  Postcode batches"):
        payload = {"postcodes": batch}
        r = requests.post(api_url, json=payload, timeout=60)
        r.raise_for_status()
        for item in r.json().get("result", []):
            res = item.get("result")
            if res:
                lsoa_code = res.get("codes", {}).get("lsoa") or res.get("lsoa")
                pc_norm = item["query"].upper().replace(" ", "")
                if lsoa_code:
                    rows.append({"pcds": pc_norm, "lsoa_code": lsoa_code})

    df = pd.DataFrame(rows)
    df.to_csv(dest, index=False)
    print(f"  Saved {len(df):,} postcode-LSOA mappings to {dest.name}")


def main(data_dir: str = "data/raw") -> None:
    """Download all raw data files.

    Parameters
    ----------
    data_dir : str, optional
        Path to raw data directory. Default is 'data/raw'.
    """
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== DfE Absence Statistics (all years) ===")
    download_all_absence_years(raw_dir)

    print("\n=== GIAS Establishments ===")
    download_gias(raw_dir)

    print("\n=== IMD 2019 ===")
    download_imd(raw_dir)

    print("\n=== ONS Postcode Directory ===")
    download_onspd(raw_dir)

    print("\nDone. Raw files in:", raw_dir.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download raw data for the school absence deprivation ML project."
    )
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    main(data_dir=args.data_dir)
