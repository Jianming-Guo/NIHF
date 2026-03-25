import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt

# =========================================================
# 0. Path Settings
# =========================================================
BASE = Path("./data")

# Your existing final tables
FINAL_DIR = BASE / "05_final_tables_v3_2"
FILE_A = FINAL_DIR / "A_seed_raw_table_v3_2.csv"
FILE_F = FINAL_DIR / "F_person_patent_dedup_table_v3_2.csv"
FILE_G = FINAL_DIR / "G_final_export_list_v3_2.csv"
FILE_G1 = FINAL_DIR / "G1_export_high_only_v3_2.csv"
FILE_G2 = FINAL_DIR / "G2_export_high_medium_v3_2.csv"

# PatentsView reference data
PV_DIR = BASE / "patentsview"
FILE_PV = PV_DIR / "Inductee_All_Patents_Detailed.csv"

# New validation output folder
OUT_DIR = BASE / "06_technical_validation_v3_2" / "patentsview_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. Configuration Parameters
# =========================================================
# Optional:
# "all"         -> Use all records from F table
# "high"        -> Validate only high
# "high_medium" -> Validate high + medium
VALIDATION_SCOPE = "high_medium"

# PatentsView only covers 1976 and later
PV_START_YEAR = 1976

# If career start point not observed, use birth year + assumed start age as fallback
ASSUMED_CAREER_START_AGE = 25
ASSUMED_CAREER_END_AGE = 70

# Image resolution
FIG_DPI = 300

# =========================================================
# 2. Utility Functions
# =========================================================
def is_missing(x):
    return pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan"

def clean_text(x):
    if is_missing(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_spaces_and_dot(s):
    s = clean_text(s)
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_name_key(x):
    """
    Loose normalization for inventor name alignment:
    - Lowercase
    - Remove dots
    - Non-alphanumeric to spaces
    - Merge consecutive spaces
    """
    if is_missing(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace(".", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_patent_id(x):
    if is_missing(x):
        return ""
    s = str(x).strip().upper()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def patent_numeric_core(x):
    """
    For example:
    US11505804B2 -> 11505804
    11505804 -> 11505804
    """
    s = clean_patent_id(x)
    nums = re.findall(r"\d+", s)
    if not nums:
        return ""
    nums = sorted(nums, key=len, reverse=True)
    return nums[0]

def extract_year_from_date(x):
    if is_missing(x):
        return np.nan
    s = str(x).strip()
    m = re.search(r"(17|18|19|20)\d{2}", s)
    if m:
        return int(m.group())
    return np.nan

def safe_min(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return vals.min() if len(vals) > 0 else np.nan

def safe_max(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return vals.max() if len(vals) > 0 else np.nan

def calc_ratio(num, den):
    if den in [0, None] or pd.isna(den):
        return np.nan
    return num / den

def pick_first_nonmissing(series):
    for x in series:
        if not is_missing(x):
            return x
    return ""

def classify_career_timing(first_year, last_year, born_year=None):
    """
    Prioritize using the first and last patent years observed in NIHF data for classification:
    - pre_1976_only
    - post_1976_start
    - spans_1976
    - unknown

    If missing, use born_year for heuristic fallback:
    - estimated_pre_1976_only
    - estimated_post_1976_start
    - estimated_spans_1976
    - unknown
    """
    fy = pd.to_numeric(first_year, errors="coerce")
    ly = pd.to_numeric(last_year, errors="coerce")
    by = pd.to_numeric(born_year, errors="coerce")

    if pd.notna(fy) and pd.notna(ly):
        if ly < PV_START_YEAR:
            return "pre_1976_only"
        elif fy >= PV_START_YEAR:
            return "post_1976_start"
        elif fy < PV_START_YEAR <= ly:
            return "spans_1976"
        else:
            return "unknown"

    if pd.notna(by):
        est_start = by + ASSUMED_CAREER_START_AGE
        est_end = by + ASSUMED_CAREER_END_AGE
        if est_end < PV_START_YEAR:
            return "estimated_pre_1976_only"
        elif est_start >= PV_START_YEAR:
            return "estimated_post_1976_start"
        elif est_start < PV_START_YEAR <= est_end:
            return "estimated_spans_1976"
        else:
            return "unknown"

    return "unknown"

def ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

# =========================================================
# 3. Load Data
# =========================================================
print("Loading data...")

A_df = pd.read_csv(FILE_A)
F_df = pd.read_csv(FILE_F)
PV_df = pd.read_csv(FILE_PV)

# Optionally read G series tables, only for scope filtering mapping
G_df = pd.read_csv(FILE_G) if FILE_G.exists() else pd.DataFrame()
G1_df = pd.read_csv(FILE_G1) if FILE_G1.exists() else pd.DataFrame()
G2_df = pd.read_csv(FILE_G2) if FILE_G2.exists() else pd.DataFrame()

# =========================================================
# 4. A Table Standardization: Construct person_key -> canonical inventor info
# =========================================================
print("Preparing A table...")

A_df = ensure_columns(
    A_df,
    ["person_key", "Inductee", "firstname", "lastname", "Born_year", "seed_patent_id_clean"]
)

A_df["canonical_inductee"] = A_df["Inductee"]
A_df["inductee_name_key"] = A_df["canonical_inductee"].apply(normalize_name_key)
A_df["born_year_clean"] = A_df["Born_year"].apply(extract_year_from_date)

A_person_lookup = (
    A_df.groupby("person_key", dropna=False)
    .agg(
        canonical_inductee=("canonical_inductee", pick_first_nonmissing),
        inductee_name_key=("inductee_name_key", pick_first_nonmissing),
        born_year_clean=("born_year_clean", safe_min),
        n_seed_rows=("person_key", "size"),
    )
    .reset_index()
)

# =========================================================
# 5. F Table Standardization: Construct NIHF inventor-patent master table for validation
# =========================================================
print("Preparing NIHF validation master table from F...")

F_df = ensure_columns(
    F_df,
    [
        "person_key",
        "pub_no_clean",
        "Publication NO.",
        "Publication Date",
        "Application Date",
        "Inductee_example",
        "best_match_level",
        "best_total_score",
        "best_seed_hit"
    ]
)

F_df["pub_no_clean"] = F_df["pub_no_clean"].apply(patent_numeric_core)
F_df["publication_year"] = F_df["Publication Date"].apply(extract_year_from_date)
F_df["application_year"] = F_df["Application Date"].apply(extract_year_from_date)
F_df["patent_year"] = F_df["application_year"]

# If application year is missing, fall back to publication year
F_df.loc[F_df["patent_year"].isna(), "patent_year"] = F_df.loc[F_df["patent_year"].isna(), "publication_year"]

# Merge inventor information from A table
NIHF_master = F_df.merge(
    A_person_lookup,
    on="person_key",
    how="left"
)

# If not matched in A, supplement with Inductee_example from F
NIHF_master["canonical_inductee"] = np.where(
    NIHF_master["canonical_inductee"].isna() | (NIHF_master["canonical_inductee"].astype(str).str.strip() == ""),
    NIHF_master["Inductee_example"],
    NIHF_master["canonical_inductee"]
)
NIHF_master["inductee_name_key"] = np.where(
    NIHF_master["inductee_name_key"].isna() | (NIHF_master["inductee_name_key"].astype(str).str.strip() == ""),
    NIHF_master["canonical_inductee"].apply(normalize_name_key),
    NIHF_master["inductee_name_key"]
)

# Remove abnormal rows without person_key or patent
NIHF_master = NIHF_master[
    NIHF_master["person_key"].notna() &
    NIHF_master["pub_no_clean"].notna() &
    (NIHF_master["pub_no_clean"].astype(str).str.strip() != "")
].copy()

# =========================================================
# 6. Filter NIHF validation scope based on scope
# =========================================================
print(f"Applying validation scope: {VALIDATION_SCOPE}")

if VALIDATION_SCOPE == "high":
    NIHf_scope_df = NIHF_master[NIHF_master["best_match_level"].astype(str).str.lower().eq("high")].copy()
elif VALIDATION_SCOPE == "high_medium":
    NIHf_scope_df = NIHF_master[
        NIHF_master["best_match_level"].astype(str).str.lower().isin(["high", "medium"])
    ].copy()
else:
    NIHf_scope_df = NIHF_master.copy()

# Keep each inventor × patent unique
NIHf_scope_df = NIHf_scope_df.drop_duplicates(subset=["person_key", "pub_no_clean"]).reset_index(drop=True)

# =========================================================
# 7. Process PatentsView data
# =========================================================
print("Preparing PatentsView table...")

PV_df = ensure_columns(PV_df, ["Inductee", "inventor_id", "patent_id"])

PV_df["inductee_name_key"] = PV_df["Inductee"].apply(normalize_name_key)
PV_df["patent_id_clean"] = PV_df["patent_id"].apply(patent_numeric_core)

PV_df = PV_df[
    PV_df["inductee_name_key"].notna() &
    (PV_df["inductee_name_key"].astype(str).str.strip() != "") &
    PV_df["patent_id_clean"].notna() &
    (PV_df["patent_id_clean"].astype(str).str.strip() != "")
].copy()

PV_df = PV_df.drop_duplicates(subset=["inductee_name_key", "inventor_id", "patent_id_clean"]).reset_index(drop=True)

# =========================================================
# 8. Construct inventor-level NIHF career statistics for career stage classification
# =========================================================
print("Building inventor-level NIHF career summary...")

career_summary = (
    NIHf_scope_df.groupby(["person_key", "inductee_name_key"], dropna=False)
    .agg(
        canonical_inductee=("canonical_inductee", pick_first_nonmissing),
        born_year_clean=("born_year_clean", safe_min),
        observed_first_patent_year=("patent_year", safe_min),
        observed_last_patent_year=("patent_year", safe_max),
        nihf_patent_n_all=("pub_no_clean", "nunique"),
        nihf_patent_n_1976plus=("pub_no_clean", lambda s: s[NIHf_scope_df.loc[s.index, "patent_year"] >= PV_START_YEAR].nunique()),
    )
    .reset_index()
)

career_summary["career_timing_group"] = career_summary.apply(
    lambda r: classify_career_timing(
        r["observed_first_patent_year"],
        r["observed_last_patent_year"],
        r["born_year_clean"]
    ),
    axis=1
)

# =========================================================
# 9. Construct NIHF 1976+ comparable window data
# =========================================================
print("Filtering NIHF patents to comparable window (1976+)...")

NIHF_1976plus = NIHf_scope_df[NIHf_scope_df["patent_year"] >= PV_START_YEAR].copy()
NIHF_1976plus = NIHF_1976plus.drop_duplicates(subset=["person_key", "pub_no_clean"]).reset_index(drop=True)

# inventors present in NIHF validation set
NIHF_inventors = career_summary[
    ["person_key", "inductee_name_key", "canonical_inductee", "career_timing_group"]
].drop_duplicates().copy()

# Only keep people in PatentsView that can be aligned with NIHF inventors
PV_comp = PV_df.merge(
    NIHF_inventors[["inductee_name_key"]].drop_duplicates(),
    on="inductee_name_key",
    how="inner"
)

# =========================================================
# 10. inventor × patent set comparison
# =========================================================
print("Comparing NIHF against PatentsView...")

# Add inventor-level information like career_timing_group to NIHF_1976plus
# Note: cannot put career_timing_group in on= because NIHF_1976plus doesn't have this column yet
NIHF_1976plus = NIHF_1976plus.merge(
    NIHF_inventors,
    on=["person_key", "inductee_name_key"],
    how="left",
    suffixes=("", "_inv")
)

# If there are duplicate columns, prioritize the original column
if "canonical_inductee_inv" in NIHF_1976plus.columns:
    NIHF_1976plus["canonical_inductee"] = np.where(
        NIHF_1976plus["canonical_inductee"].isna() |
        (NIHF_1976plus["canonical_inductee"].astype(str).str.strip() == ""),
        NIHF_1976plus["canonical_inductee_inv"],
        NIHF_1976plus["canonical_inductee"]
    )
    NIHF_1976plus.drop(columns=["canonical_inductee_inv"], inplace=True)

# inventor-level patent sets
nihf_sets = (
    NIHF_1976plus.groupby(["person_key", "inductee_name_key"], dropna=False)["pub_no_clean"]
    .apply(lambda s: set(s.dropna().astype(str)))
    .reset_index(name="nihf_1976plus_patent_set")
)

pv_sets = (
    PV_comp.groupby("inductee_name_key", dropna=False)["patent_id_clean"]
    .apply(lambda s: set(s.dropna().astype(str)))
    .reset_index(name="pv_patent_set")
)

comparison = career_summary.merge(
    nihf_sets,
    on=["person_key", "inductee_name_key"],
    how="left"
).merge(
    pv_sets,
    on="inductee_name_key",
    how="left"
)

comparison["nihf_1976plus_patent_set"] = comparison["nihf_1976plus_patent_set"].apply(
    lambda x: x if isinstance(x, set) else set()
)
comparison["pv_patent_set"] = comparison["pv_patent_set"].apply(
    lambda x: x if isinstance(x, set) else set()
)

comparison["intersection_set"] = comparison.apply(
    lambda r: r["nihf_1976plus_patent_set"] & r["pv_patent_set"], axis=1
)
comparison["pv_only_set"] = comparison.apply(
    lambda r: r["pv_patent_set"] - r["nihf_1976plus_patent_set"], axis=1
)
comparison["nihf_extra_set"] = comparison.apply(
    lambda r: r["nihf_1976plus_patent_set"] - r["pv_patent_set"], axis=1
)

comparison["pv_patent_n"] = comparison["pv_patent_set"].apply(len)
comparison["nihf_patent_n_1976plus"] = comparison["nihf_1976plus_patent_set"].apply(len)
comparison["intersection_n"] = comparison["intersection_set"].apply(len)
comparison["pv_only_n"] = comparison["pv_only_set"].apply(len)
comparison["nihf_extra_n"] = comparison["nihf_extra_set"].apply(len)

comparison["coverage_vs_pv"] = comparison.apply(
    lambda r: calc_ratio(r["intersection_n"], r["pv_patent_n"]),
    axis=1
)
comparison["agreement_vs_nihf"] = comparison.apply(
    lambda r: calc_ratio(r["intersection_n"], r["nihf_patent_n_1976plus"]),
    axis=1
)
comparison["jaccard_similarity"] = comparison.apply(
    lambda r: calc_ratio(
        r["intersection_n"],
        len(r["pv_patent_set"] | r["nihf_1976plus_patent_set"])
    ),
    axis=1
)

# Whether fully recovered in PatentsView
comparison["pv_fully_recovered"] = comparison.apply(
    lambda r: (r["pv_patent_n"] > 0) and (r["intersection_n"] == r["pv_patent_n"]),
    axis=1
)

# =========================================================
# 11. Generate detail tables: missing / extra / matched
# =========================================================
print("Building patent-level detail tables...")

missing_rows = []
extra_rows = []
matched_rows = []

for _, row in comparison.iterrows():
    person_key = row["person_key"]
    name_key = row["inductee_name_key"]
    canonical_inductee = row["canonical_inductee"]
    career_group = row["career_timing_group"]

    for p in sorted(row["pv_only_set"]):
        missing_rows.append({
            "person_key": person_key,
            "inductee_name_key": name_key,
            "canonical_inductee": canonical_inductee,
            "career_timing_group": career_group,
            "patent_id_clean": p,
            "detail_type": "pv_only_missing_in_nihf"
        })

    for p in sorted(row["nihf_extra_set"]):
        extra_rows.append({
            "person_key": person_key,
            "inductee_name_key": name_key,
            "canonical_inductee": canonical_inductee,
            "career_timing_group": career_group,
            "patent_id_clean": p,
            "detail_type": "nihf_extra_not_in_pv"
        })

    for p in sorted(row["intersection_set"]):
        matched_rows.append({
            "person_key": person_key,
            "inductee_name_key": name_key,
            "canonical_inductee": canonical_inductee,
            "career_timing_group": career_group,
            "patent_id_clean": p,
            "detail_type": "matched_in_both"
        })

missing_detail_df = pd.DataFrame(missing_rows)
extra_detail_df = pd.DataFrame(extra_rows)
matched_detail_df = pd.DataFrame(matched_rows)

# Try to supplement more fields from NIHF into extra/matched details
nihf_patent_lookup_cols = [
    "person_key", "pub_no_clean", "Publication NO.", "Publication Date", "Application Date",
    "patent_year", "best_match_level", "best_total_score", "best_seed_hit"
]
nihf_patent_lookup_cols = [c for c in nihf_patent_lookup_cols if c in NIHF_1976plus.columns]
nihf_patent_lookup = NIHF_1976plus[nihf_patent_lookup_cols].drop_duplicates(
    subset=["person_key", "pub_no_clean"]
).rename(columns={"pub_no_clean": "patent_id_clean"})

if not extra_detail_df.empty:
    extra_detail_df = extra_detail_df.merge(
        nihf_patent_lookup,
        on=["person_key", "patent_id_clean"],
        how="left"
    )

if not matched_detail_df.empty:
    matched_detail_df = matched_detail_df.merge(
        nihf_patent_lookup,
        on=["person_key", "patent_id_clean"],
        how="left"
    )

# =========================================================
# 12. Overall Summary Table
# =========================================================
print("Building overall summary...")

overall_summary_rows = []

comparison_valid = comparison[comparison["pv_patent_n"] > 0].copy()

n_inventors_total = len(comparison)
n_inventors_with_pv = len(comparison_valid)
n_inventors_with_nihf_1976plus_records = int((comparison_valid["nihf_patent_n_1976plus"] > 0).sum())
n_inventors_fully_recovered = int(comparison_valid["pv_fully_recovered"].sum())

total_pv_patents = int(comparison_valid["pv_patent_n"].sum())
total_nihf_1976plus_patents = int(comparison_valid["nihf_patent_n_1976plus"].sum())
total_intersection = int(comparison_valid["intersection_n"].sum())
total_pv_only = int(comparison_valid["pv_only_n"].sum())
total_nihf_extra = int(comparison_valid["nihf_extra_n"].sum())

mean_inventor_coverage = comparison_valid["coverage_vs_pv"].mean()
median_inventor_coverage = comparison_valid["coverage_vs_pv"].median()
mean_inventor_agreement = comparison_valid["agreement_vs_nihf"].mean()
median_inventor_agreement = comparison_valid["agreement_vs_nihf"].median()
mean_inventor_jaccard = comparison_valid["jaccard_similarity"].mean()
median_inventor_jaccard = comparison_valid["jaccard_similarity"].median()



total_intersection = int(comparison["intersection_n"].sum())
total_pv_only = int(comparison["pv_only_n"].sum())
total_nihf_extra = int(comparison["nihf_extra_n"].sum())

overall_summary_rows.extend([
    {"metric": "validation_scope", "value": VALIDATION_SCOPE},
    {"metric": "pv_start_year", "value": PV_START_YEAR},
    {"metric": "n_inventors_total", "value": n_inventors_total},
    {"metric": "n_inventors_with_pv_records", "value": n_inventors_with_pv},
    {"metric": "n_inventors_with_nihf_1976plus_records", "value": n_inventors_with_nihf_1976plus_records},
    {"metric": "n_inventors_fully_recovered_in_pv", "value": n_inventors_fully_recovered},
    {"metric": "total_pv_patents", "value": total_pv_patents},
    {"metric": "total_nihf_1976plus_patents", "value": total_nihf_1976plus_patents},
    {"metric": "total_intersection_patents", "value": total_intersection},
    {"metric": "total_pv_only_patents", "value": total_pv_only},
    {"metric": "total_nihf_extra_patents", "value": total_nihf_extra},
    {"metric": "pooled_coverage_vs_pv", "value": calc_ratio(total_intersection, total_pv_patents)},
    {"metric": "pooled_agreement_vs_nihf", "value": calc_ratio(total_intersection, total_nihf_1976plus_patents)},
    {
        "metric": "pooled_jaccard_similarity",
        "value": calc_ratio(total_intersection, total_pv_patents + total_nihf_1976plus_patents - total_intersection)
    },
    {"metric": "mean_inventor_coverage_vs_pv", "value": comparison["coverage_vs_pv"].mean()},
    {"metric": "median_inventor_coverage_vs_pv", "value": comparison["coverage_vs_pv"].median()},
    {"metric": "mean_inventor_agreement_vs_nihf", "value": comparison["agreement_vs_nihf"].mean()},
    {"metric": "median_inventor_agreement_vs_nihf", "value": comparison["agreement_vs_nihf"].median()},
    {"metric": "mean_inventor_jaccard", "value": comparison["jaccard_similarity"].mean()},
    {"metric": "median_inventor_jaccard", "value": comparison["jaccard_similarity"].median()},
])

overall_summary_df = pd.DataFrame(overall_summary_rows)

# =========================================================
# 13. Group Summary: career timing
# =========================================================
print("Building group summary...")

group_summary = (
    comparison_valid.groupby("career_timing_group", dropna=False)
    .agg(
        n_inventors=("person_key", "nunique"),
        inventors_with_pv=("pv_patent_n", lambda s: int((s > 0).sum())),
        total_pv_patents=("pv_patent_n", "sum"),
        total_nihf_1976plus_patents=("nihf_patent_n_1976plus", "sum"),
        total_intersection=("intersection_n", "sum"),
        total_pv_only=("pv_only_n", "sum"),
        total_nihf_extra=("nihf_extra_n", "sum"),
        mean_coverage_vs_pv=("coverage_vs_pv", "mean"),
        median_coverage_vs_pv=("coverage_vs_pv", "median"),
        mean_agreement_vs_nihf=("agreement_vs_nihf", "mean"),
        median_agreement_vs_nihf=("agreement_vs_nihf", "median"),
        mean_jaccard=("jaccard_similarity", "mean"),
        median_jaccard=("jaccard_similarity", "median"),
    )
    .reset_index()
)

group_summary["pooled_coverage_vs_pv"] = group_summary.apply(
    lambda r: calc_ratio(r["total_intersection"], r["total_pv_patents"]),
    axis=1
)
group_summary["pooled_agreement_vs_nihf"] = group_summary.apply(
    lambda r: calc_ratio(r["total_intersection"], r["total_nihf_1976plus_patents"]),
    axis=1
)
group_summary["pooled_jaccard"] = group_summary.apply(
    lambda r: calc_ratio(
        r["total_intersection"],
        r["total_pv_patents"] + r["total_nihf_1976plus_patents"] - r["total_intersection"]
    ),
    axis=1
)

# =========================================================
# 14. Supplement: Record main missing inventors for paper description
# =========================================================
print("Building low-coverage inventor list...")

low_coverage_df = comparison.sort_values(
    by=["coverage_vs_pv", "pv_patent_n", "canonical_inductee"],
    ascending=[True, False, True]
).reset_index(drop=True)

# =========================================================
# 15. Export CSV
# =========================================================
print("Saving CSV outputs...")

NIHF_master.to_csv(OUT_DIR / "01_nihf_validation_master_table.csv", index=False, encoding="utf-8-sig")
NIHF_1976plus.to_csv(OUT_DIR / "02_nihf_1976plus_comparable_table.csv", index=False, encoding="utf-8-sig")
PV_comp.to_csv(OUT_DIR / "03_patentsview_comparable_table.csv", index=False, encoding="utf-8-sig")

career_summary.to_csv(OUT_DIR / "04_nihf_inventor_career_summary.csv", index=False, encoding="utf-8-sig")
comparison.to_csv(OUT_DIR / "05_inventor_level_patentsview_comparison.csv", index=False, encoding="utf-8-sig")

missing_detail_df.to_csv(OUT_DIR / "06_patents_missing_in_nihf_but_present_in_patentsview.csv", index=False, encoding="utf-8-sig")
extra_detail_df.to_csv(OUT_DIR / "07_patents_present_in_nihf_but_not_in_patentsview.csv", index=False, encoding="utf-8-sig")
matched_detail_df.to_csv(OUT_DIR / "08_patents_matched_in_both_datasets.csv", index=False, encoding="utf-8-sig")

overall_summary_df.to_csv(OUT_DIR / "09_overall_validation_summary.csv", index=False, encoding="utf-8-sig")
group_summary.to_csv(OUT_DIR / "10_validation_summary_by_career_timing_group.csv", index=False, encoding="utf-8-sig")
low_coverage_df.to_csv(OUT_DIR / "11_low_coverage_inventor_list.csv", index=False, encoding="utf-8-sig")

# =========================================================
# 16. Simple Visualization
# =========================================================
print("Creating figures...")

# Fig 1: inventor-level coverage histogram
fig, ax = plt.subplots(figsize=(7, 5))
plot_series = comparison["coverage_vs_pv"].dropna()
if len(plot_series) > 0:
    ax.hist(plot_series, bins=20)
ax.set_xlabel("Inventor-level coverage vs PatentsView")
ax.set_ylabel("Number of inventors")
ax.set_title("Coverage distribution across inventors")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_01_coverage_histogram.png", dpi=FIG_DPI)
plt.close()

# Fig 2: NIHF 1976+ vs PV patent counts
fig, ax = plt.subplots(figsize=(6, 6))
x = comparison["pv_patent_n"].fillna(0)
y = comparison["nihf_patent_n_1976plus"].fillna(0)
ax.scatter(x, y, alpha=0.7)
max_xy = max(x.max() if len(x) else 0, y.max() if len(y) else 0)
ax.plot([0, max_xy], [0, max_xy], linewidth=1)
ax.set_xlabel("PatentsView patent count")
ax.set_ylabel("NIHF patent count (1976+)")
ax.set_title("Patent counts by inventor")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_02_patent_count_scatter.png", dpi=FIG_DPI)
plt.close()

# Fig 3: career group pooled coverage
group_plot = group_summary.copy()
group_plot = group_plot.sort_values("career_timing_group").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(group_plot["career_timing_group"].astype(str), group_plot["pooled_coverage_vs_pv"])
ax.set_xlabel("Career timing group")
ax.set_ylabel("Pooled coverage vs PatentsView")
ax.set_title("Coverage by career timing group")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_03_group_pooled_coverage.png", dpi=FIG_DPI)
plt.close()

# =========================================================
# 17. Save Log
# =========================================================
print("Saving run log...")

run_log = pd.DataFrame([
    {"item": "VALIDATION_SCOPE", "value": VALIDATION_SCOPE},
    {"item": "PV_START_YEAR", "value": PV_START_YEAR},
    {"item": "ASSUMED_CAREER_START_AGE", "value": ASSUMED_CAREER_START_AGE},
    {"item": "ASSUMED_CAREER_END_AGE", "value": ASSUMED_CAREER_END_AGE},
    {"item": "A_rows", "value": len(A_df)},
    {"item": "F_rows", "value": len(F_df)},
    {"item": "PV_rows", "value": len(PV_df)},
    {"item": "NIHF_master_rows", "value": len(NIHF_master)},
    {"item": "NIHF_1976plus_rows", "value": len(NIHF_1976plus)},
    {"item": "comparison_inventors", "value": len(comparison)},
])

run_log.to_csv(OUT_DIR / "run_log_patentsview_comparison.csv", index=False, encoding="utf-8-sig")

print("\n=== best_match_level distribution in F ===")
print(F_df["best_match_level"].value_counts(dropna=False))

print("\n=== best_match_level distribution in NIHF_master ===")
print(NIHF_master["best_match_level"].value_counts(dropna=False))

print("\n=== best_match_level distribution in NIHf_scope_df before 1976 filter ===")
print(NIHf_scope_df["best_match_level"].value_counts(dropna=False))

tmp_1976 = NIHf_scope_df[NIHf_scope_df["patent_year"] >= PV_START_YEAR].copy()
print("\n=== best_match_level distribution in NIHf_scope_df (1976+) ===")
print(tmp_1976["best_match_level"].value_counts(dropna=False))

print("Done.")
print(f"Validation results saved to: {OUT_DIR}")