# -*- coding: utf-8 -*-
"""
Build release-ready dataset for NIHF patent linkage project (v3.2)

Output folder:
./data/08_release_dataset_v3_2/

Generated files:
1. 01_nihf_inventors_release_v3_2.csv
2. 02_nihf_inventor_patent_links_high_medium_release_v3_2.csv
3. 03_nihf_inventor_patent_links_all_release_v3_2.csv
4. 04_nihf_data_dictionary_release_v3_2.csv
5. README_release_structure_v3_2.txt
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# 0. Paths
# =========================================================
BASE = Path("./data")
INPUT_DIR = BASE / "05_final_tables_v3_2"
OUT_DIR = BASE / "08_release_dataset_v3_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_A = INPUT_DIR / "A_seed_raw_table_v3_2.csv"
FILE_F = INPUT_DIR / "F_person_patent_dedup_table_v3_2.csv"

# =========================================================
# 1. Helper functions
# =========================================================
def safe_read_csv(path: Path) -> pd.DataFrame:
    encodings_to_try = ["utf-8-sig", "utf-8", "gb18030", "latin1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_error = e
    raise last_error


def ensure_columns(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def is_missing(x):
    return pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan"


def extract_year(x):
    """
    Extract year from mixed date formats, including:
    - yyyymmdd (e.g., 19520101)
    - yyyy/mm/dd
    - yyyy-mm-dd
    - plain yyyy
    """
    if is_missing(x):
        return pd.NA

    s = str(x).strip()

    # Handle float-like strings such as 19520101.0
    if s.endswith(".0"):
        s = s[:-2]

    # Prefer explicit 8-digit yyyymmdd starting with 17/18/19/20
    m8 = re.fullmatch(r"(17|18|19|20)\d{6}", s)
    if m8:
        return int(s[:4])

    # Generic year extraction
    m = re.search(r"(17|18|19|20)\d{2}", s)
    if m:
        return int(m.group())

    return pd.NA


def patent_numeric_core(x):
    """
    Example:
    US11505804B2 -> 11505804
    11505804 -> 11505804
    """
    if is_missing(x):
        return ""
    s = str(x).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^A-Z0-9]", "", s)
    nums = re.findall(r"\d+", s)
    if not nums:
        return ""
    nums = sorted(nums, key=len, reverse=True)
    return nums[0]


def choose_patent_id(pub_no, pub_no_clean):
    """
    Prefer formatted publication number if available, else fallback to cleaned numeric core.
    """
    if not is_missing(pub_no):
        return str(pub_no).strip()
    if not is_missing(pub_no_clean):
        return str(pub_no_clean).strip()
    return ""


def first_nonmissing(series):
    for x in series:
        if not is_missing(x):
            return x
    return ""


def count_unique_nonmissing(series):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s != ""]
    return s.nunique()


def join_unique_nonmissing(series, sep=";"):
    s = pd.Series(series).dropna().astype(str).str.strip()
    s = s[s != ""]
    vals = list(dict.fromkeys(s.tolist()))  # preserve order, remove duplicates
    return sep.join(vals)


# =========================================================
# 2. Load data
# =========================================================
print("Loading A and F tables...")

A = safe_read_csv(FILE_A)
F = safe_read_csv(FILE_F)

# =========================================================
# 3. Prepare inventor release table (Table 01)
# =========================================================
print("Preparing inventor release table...")

A = ensure_columns(
    A,
    [
        "person_key", "Inductee", "firstname", "lastname", "Tech",
        "patent_id", "seed_patent_id", "seed_patent_id_clean",
        "Inducted_year", "Born_year", "Died_year"
    ]
)

# Standardize year fields
A["Inducted_year_clean"] = A["Inducted_year"].apply(extract_year)
A["Born_year_clean"] = A["Born_year"].apply(extract_year)
A["Died_year_clean"] = A["Died_year"].apply(extract_year)

# Build seed patent field for counting and release display
A["seed_patent_for_count"] = A["seed_patent_id_clean"]

mask_missing_seed = A["seed_patent_for_count"].isna() | (
    A["seed_patent_for_count"].astype(str).str.strip() == ""
)
A.loc[mask_missing_seed, "seed_patent_for_count"] = A.loc[mask_missing_seed, "patent_id"]

# Keep a release-style patent_id field for table 01
# Here patent_id represents the seed patent identifiers associated with the inventor
A["seed_patent_for_release"] = A["seed_patent_id_clean"]

mask_missing_release = A["seed_patent_for_release"].isna() | (
    A["seed_patent_for_release"].astype(str).str.strip() == ""
)
A.loc[mask_missing_release, "seed_patent_for_release"] = A.loc[mask_missing_release, "patent_id"]

inventor_release = (
    A.groupby("person_key", dropna=False)
    .agg(
        Inductee=("Inductee", first_nonmissing),
        firstname=("firstname", first_nonmissing),
        lastname=("lastname", first_nonmissing),
        tech_domain=("Tech", first_nonmissing),
        patent_id=("seed_patent_for_release", join_unique_nonmissing),
        n_seed_patents=("seed_patent_for_count", count_unique_nonmissing),
        Inducted_year=("Inducted_year_clean", "min"),
        Born_year=("Born_year_clean", "min"),
        Died_year=("Died_year_clean", "min"),
    )
    .reset_index()
)

# Stable public inventor_id
inventor_release = inventor_release.sort_values(
    by=["Inductee", "person_key"],
    kind="stable"
).reset_index(drop=True)

inventor_release["inventor_id"] = [
    f"nihf_{i:04d}" for i in range(1, len(inventor_release) + 1)
]

inventor_release = inventor_release[
    [
        "inventor_id",
        "person_key",
        "Inductee",
        "firstname",
        "lastname",
        "tech_domain",
        "patent_id",
        "n_seed_patents",
        "Inducted_year",
        "Born_year",
        "Died_year",
    ]
].copy()

# Nullable integer columns
for c in ["n_seed_patents", "Inducted_year", "Born_year", "Died_year"]:
    inventor_release[c] = pd.to_numeric(inventor_release[c], errors="coerce").astype("Int64")

# =========================================================
# 4. Prepare linkage tables (Table 02 and 03)
# =========================================================
print("Preparing inventor–patent linkage release tables...")

F = ensure_columns(
    F,
    [
        "person_key", "pub_no_clean", "Publication NO.",
        "Publication Date", "Application Date",
        "Inductee_example", "best_match_level", "best_total_score", "best_seed_hit",
        "n_supporting_rows", "n_high_support", "n_medium_support", "n_low_support"
    ]
)

F["patent_id_clean"] = F["pub_no_clean"].apply(patent_numeric_core)
F["patent_id"] = F.apply(
    lambda r: choose_patent_id(r["Publication NO."], r["patent_id_clean"]),
    axis=1
)
F["application_year"] = F["Application Date"].apply(extract_year)
F["publication_year"] = F["Publication Date"].apply(extract_year)
F["match_level"] = F["best_match_level"].astype(str).str.strip().str.lower()
F["match_score"] = pd.to_numeric(F["best_total_score"], errors="coerce")
F["seed_hit"] = pd.to_numeric(F["best_seed_hit"], errors="coerce")

for c in ["n_supporting_rows", "n_high_support", "n_medium_support", "n_low_support"]:
    F[c] = pd.to_numeric(F[c], errors="coerce").astype("Int64")

# Merge inventor_id + inventor name
link_base = F.merge(
    inventor_release[["inventor_id", "person_key", "Inductee"]],
    on="person_key",
    how="left"
)

# inventor_name field for release linkage table
link_base["inventor_name"] = np.where(
    link_base["Inductee"].isna() | (link_base["Inductee"].astype(str).str.strip() == ""),
    link_base["Inductee_example"],
    link_base["Inductee"]
)

# Keep only clean rows
link_base = link_base[
    link_base["person_key"].notna() &
    link_base["patent_id_clean"].notna() &
    (link_base["patent_id_clean"].astype(str).str.strip() != "")
].copy()

# Deduplicate release rows just in case
link_base = link_base.drop_duplicates(subset=["person_key", "patent_id_clean"]).copy()

link_release_all = link_base[
    [
        "inventor_id", "person_key", "inventor_name",
        "patent_id", "patent_id_clean",
        "application_year", "publication_year",
        "match_level", "match_score", "seed_hit",
        "n_supporting_rows", "n_high_support", "n_medium_support", "n_low_support"
    ]
].copy()

# Recommended release: high + medium
link_release_hm = link_release_all[
    link_release_all["match_level"].isin(["high", "medium"])
].copy()

# Nullable integer formatting
for df_ in [link_release_all, link_release_hm]:
    for c in [
        "application_year", "publication_year",
        "n_supporting_rows", "n_high_support",
        "n_medium_support", "n_low_support"
    ]:
        df_[c] = pd.to_numeric(df_[c], errors="coerce").astype("Int64")

# Sort for reproducibility
sort_cols = ["inventor_id", "patent_id_clean"]
link_release_all = link_release_all.sort_values(sort_cols, kind="stable").reset_index(drop=True)
link_release_hm = link_release_hm.sort_values(sort_cols, kind="stable").reset_index(drop=True)

# =========================================================
# 5. Build data dictionary
# =========================================================
print("Preparing data dictionary...")

dictionary_rows = [
    # Table 01
    ["01_nihf_inventors_release_v3_2.csv", "inventor_id", "string", "Public unique identifier for each NIHF inventor.", "nihf_0001"],
    ["01_nihf_inventors_release_v3_2.csv", "person_key", "string", "Internal inventor identifier used in the matching pipeline.", "adi||shamir||1952"],
    ["01_nihf_inventors_release_v3_2.csv", "Inductee", "string", "Full name of the NIHF inventor.", "Adi Shamir"],
    ["01_nihf_inventors_release_v3_2.csv", "firstname", "string", "Inventor first name.", "Adi"],
    ["01_nihf_inventors_release_v3_2.csv", "lastname", "string", "Inventor last name.", "Shamir"],
    ["01_nihf_inventors_release_v3_2.csv", "tech_domain", "string", "Technology domain recorded in NIHF source data when available.", "Cryptography"],
    ["01_nihf_inventors_release_v3_2.csv", "patent_id", "string", "Seed patent identifier(s) associated with the inventor, joined by semicolons if multiple.", "4405829"],
    ["01_nihf_inventors_release_v3_2.csv", "n_seed_patents", "Int64", "Number of unique seed patents associated with the inventor.", "1"],
    ["01_nihf_inventors_release_v3_2.csv", "Inducted_year", "Int64", "Year of induction into the National Inventors Hall of Fame.", "2017"],
    ["01_nihf_inventors_release_v3_2.csv", "Born_year", "Int64", "Inventor birth year extracted from NIHF source records.", "1952"],
    ["01_nihf_inventors_release_v3_2.csv", "Died_year", "Int64", "Inventor death year extracted from NIHF source records when available.", ""],

    # Table 02
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "inventor_id", "string", "Public unique identifier for each NIHF inventor.", "nihf_0001"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "person_key", "string", "Internal inventor identifier used in the matching pipeline.", "adi||shamir||1952"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "inventor_name", "string", "Full inventor name.", "Adi Shamir"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "patent_id", "string", "Patent publication identifier, preferably in formatted form.", "US4405829A"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "patent_id_clean", "string", "Cleaned patent identifier using the numeric patent core.", "4405829"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "application_year", "Int64", "Patent application year when available.", "1977"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "publication_year", "Int64", "Patent publication year when available.", "1983"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "match_level", "string", "Confidence level of inventor–patent linkage.", "high"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "match_score", "float", "Aggregated matching score across multiple evidence dimensions.", "160"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "seed_hit", "float", "Seed support score or seed-hit contribution in the matching framework.", "25"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "n_supporting_rows", "Int64", "Number of supporting matched rows merged into the final inventor–patent linkage.", "1"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "n_high_support", "Int64", "Number of supporting rows classified as high-support.", "1"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "n_medium_support", "Int64", "Number of supporting rows classified as medium-support.", "0"],
    ["02_nihf_inventor_patent_links_high_medium_release_v3_2.csv", "n_low_support", "Int64", "Number of supporting rows classified as low-support.", "0"],

    # Table 03
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "inventor_id", "string", "Public unique identifier for each NIHF inventor.", "nihf_0001"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "person_key", "string", "Internal inventor identifier used in the matching pipeline.", "adi||shamir||1952"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "inventor_name", "string", "Full inventor name.", "Adi Shamir"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "patent_id", "string", "Patent publication identifier, preferably in formatted form.", "US4405829A"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "patent_id_clean", "string", "Cleaned patent identifier using the numeric patent core.", "4405829"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "application_year", "Int64", "Patent application year when available.", "1977"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "publication_year", "Int64", "Patent publication year when available.", "1983"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "match_level", "string", "Confidence level of inventor–patent linkage.", "low"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "match_score", "float", "Aggregated matching score across multiple evidence dimensions.", "45"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "seed_hit", "float", "Seed support score or seed-hit contribution in the matching framework.", "0"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "n_supporting_rows", "Int64", "Number of supporting matched rows merged into the final inventor–patent linkage.", "1"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "n_high_support", "Int64", "Number of supporting rows classified as high-support.", "0"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "n_medium_support", "Int64", "Number of supporting rows classified as medium-support.", "0"],
    ["03_nihf_inventor_patent_links_all_release_v3_2.csv", "n_low_support", "Int64", "Number of supporting rows classified as low-support.", "1"],
]

data_dictionary = pd.DataFrame(
    dictionary_rows,
    columns=["file_name", "variable_name", "data_type", "description", "example"]
)

# =========================================================
# 6. Write README
# =========================================================
print("Writing README...")

readme_text = """NIHF release dataset v3.2

Files
-----
01_nihf_inventors_release_v3_2.csv
    Inventor-level metadata for NIHF inductees, including seed patent identifiers.

02_nihf_inventor_patent_links_high_medium_release_v3_2.csv
    Recommended release version containing high- and medium-confidence inventor–patent linkages.

03_nihf_inventor_patent_links_all_release_v3_2.csv
    Full linkage layer containing high-, medium-, and low-confidence inventor–patent linkages.

04_nihf_data_dictionary_release_v3_2.csv
    Variable-level data dictionary for the release files.

Recommended usage
-----------------
Use 02_nihf_inventor_patent_links_high_medium_release_v3_2.csv
for most downstream analyses.

The all-release file is provided for transparency, methodological
reproducibility, and exploratory analysis.

Notes
-----
This release focuses on inventor–patent linkage data rather than
full patent metadata. Additional patent-level variables such as
citations, classifications, assignees, and text fields can be
retrieved by linking the provided patent identifiers to external
patent databases.
"""

# =========================================================
# 7. Save files
# =========================================================
print("Saving release files...")

inventor_release.to_csv(
    OUT_DIR / "01_nihf_inventors_release_v3_2.csv",
    index=False,
    encoding="utf-8-sig"
)

link_release_hm.to_csv(
    OUT_DIR / "02_nihf_inventor_patent_links_high_medium_release_v3_2.csv",
    index=False,
    encoding="utf-8-sig"
)

link_release_all.to_csv(
    OUT_DIR / "03_nihf_inventor_patent_links_all_release_v3_2.csv",
    index=False,
    encoding="utf-8-sig"
)

data_dictionary.to_csv(
    OUT_DIR / "04_nihf_data_dictionary_release_v3_2.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(OUT_DIR / "README_release_structure_v3_2.txt", "w", encoding="utf-8") as f:
    f.write(readme_text)

# =========================================================
# 8. Print summary
# =========================================================
print("\n=== Release dataset summary ===")
print(f"Inventor table rows: {len(inventor_release):,}")
print(f"High+Medium linkage rows: {len(link_release_hm):,}")
print(f"All linkage rows: {len(link_release_all):,}")
print(f"Unique patents in high+medium: {link_release_hm['patent_id_clean'].nunique():,}")
print(f"Unique patents in all: {link_release_all['patent_id_clean'].nunique():,}")

print("\n=== Check n_seed_patents distribution ===")
print(inventor_release["n_seed_patents"].describe())
print(inventor_release["n_seed_patents"].value_counts(dropna=False).head(20))

print(f"\nOutput folder: {OUT_DIR}")
print("Done.")