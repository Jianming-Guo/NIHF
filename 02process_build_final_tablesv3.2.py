import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================================================
# 0. Path Settings
# =========================================================
BASE = Path("./data")

FILE_SEED_RAW = BASE / "NIHF_patent_matched.csv"
FILE_SEED_ENRICH_SOURCE = BASE / "NIHF_raw" / "NIHF_raw_download.xlsx"

MERGED_DIR = BASE / "04_match_results_v3_2" / "merged"
FILE_ALL_DOWNLOAD_RAW = MERGED_DIR / "all_download_raw_merged_v3_2.csv"
FILE_ALL_CANDIDATE_MATCHES = MERGED_DIR / "all_candidate_matches_v3_2.csv"
FILE_ALL_MATCH_SUMMARY = MERGED_DIR / "all_match_summary_v3_2.csv"

OUT_DIR = BASE / "05_final_tables_v3_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. Utility Functions
# =========================================================
def is_missing(x):
    return pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan"

def clean_text(x):
    if is_missing(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
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
    s = clean_patent_id(x)
    nums = re.findall(r"\d+", s)
    if not nums:
        return ""
    nums = sorted(nums, key=len, reverse=True)
    return nums[0]

def extract_year_from_date(x):
    if is_missing(x):
        return None
    s = str(x).strip()
    m = re.search(r"(17|18|19|20)\d{2}", s)
    if m:
        return int(m.group())
    return None

def normalize_spaces_and_dot(s):
    s = clean_text(s)
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_person_key(firstname, lastname, born_year):
    fn = normalize_spaces_and_dot(firstname)
    ln = normalize_spaces_and_dot(lastname)
    by = extract_year_from_date(born_year)
    if by is not None:
        return f"{fn}||{ln}||{by}"
    return f"{fn}||{ln}"

def safe_join(seq, sep=" | "):
    out = []
    for x in seq:
        if pd.isna(x):
            continue
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            continue
        out.append(s)
    return sep.join(out)

def best_level_from_series(levels):
    levels = set([str(x).strip().lower() for x in levels if not is_missing(x)])
    if "high" in levels:
        return "high"
    if "medium" in levels:
        return "medium"
    if "low" in levels:
        return "low"
    return ""

# =========================================================
# 2. Load Data
# =========================================================
print("Loading source files...")

seed_raw_df = pd.read_csv(FILE_SEED_RAW)
seed_enrich_source_df = pd.read_excel(FILE_SEED_ENRICH_SOURCE)

all_download_raw_df = pd.read_csv(FILE_ALL_DOWNLOAD_RAW)
all_candidate_matches_df = pd.read_csv(FILE_ALL_CANDIDATE_MATCHES)
all_match_summary_df = pd.read_csv(FILE_ALL_MATCH_SUMMARY)

# =========================================================
# 3. Table A: Original Seed Table
# =========================================================
print("Building Table A...")

A_df = seed_raw_df.copy()

# Add standardized key
A_df["person_key"] = A_df.apply(
    lambda r: build_person_key(r.get("firstname", ""), r.get("lastname", ""), r.get("Born_year", "")),
    axis=1
)
A_df["seed_patent_id_clean"] = A_df["patent_id"].apply(patent_numeric_core)

# Suggested sorting
sort_cols_A = [c for c in ["No", "person_key", "seed_patent_id_clean"] if c in A_df.columns]
if sort_cols_A:
    A_df = A_df.sort_values(by=sort_cols_A).reset_index(drop=True)

A_path = OUT_DIR / "A_seed_raw_table_v3_2.csv"
A_df.to_csv(A_path, index=False, encoding="utf-8-sig")

# =========================================================
# 4. Table B: Seed Patent Enrichment Table
# =========================================================
print("Building Table B...")

B_source = seed_enrich_source_df.copy()
B_source["seed_patent_id_clean"] = B_source["Publication Number"].apply(patent_numeric_core)

# Merge using seed_patent_id_clean from Table A
B_df = A_df.merge(
    B_source,
    on="seed_patent_id_clean",
    how="left",
    suffixes=("", "_seedinfo")
)

# Add a flag indicating if enrichment was successful
B_df["seed_patent_info_found"] = B_df["Publication Number"].notna()

sort_cols_B = [c for c in ["No", "person_key", "seed_patent_id_clean"] if c in B_df.columns]
if sort_cols_B:
    B_df = B_df.sort_values(by=sort_cols_B).reset_index(drop=True)

B_path = OUT_DIR / "B_seed_patent_enriched_table_v3_2.csv"
B_df.to_csv(B_path, index=False, encoding="utf-8-sig")

# =========================================================
# 5. Table C: All Candidate Raw Table
# =========================================================
print("Building Table C...")

C_df = all_download_raw_df.copy()

# If standardized key not present, add it here
if "pub_no_clean" not in C_df.columns:
    if "Publication NO." in C_df.columns:
        C_df["pub_no_clean"] = C_df["Publication NO."].apply(patent_numeric_core)
    elif "Publication Number" in C_df.columns:
        C_df["pub_no_clean"] = C_df["Publication Number"].apply(patent_numeric_core)

sort_cols_C = [c for c in ["group_id", "pub_no_clean"] if c in C_df.columns]
if sort_cols_C:
    C_df = C_df.sort_values(by=sort_cols_C).reset_index(drop=True)

C_path = OUT_DIR / "C_all_candidate_raw_table_v3_2.csv"
C_df.to_csv(C_path, index=False, encoding="utf-8-sig")

# =========================================================
# 6. Table D: Inventor-Candidate Patent Match Detail Table
# =========================================================
print("Building Table D...")

D_df = all_candidate_matches_df.copy()

# Defensive key addition
if "person_key" not in D_df.columns:
    D_df["person_key"] = D_df.apply(
        lambda r: build_person_key(r.get("firstname", ""), r.get("lastname", ""), r.get("Born_year", "")),
        axis=1
    )

if "pub_no_clean" not in D_df.columns:
    if "Publication NO." in D_df.columns:
        D_df["pub_no_clean"] = D_df["Publication NO."].apply(patent_numeric_core)
    elif "Publication Number" in D_df.columns:
        D_df["pub_no_clean"] = D_df["Publication Number"].apply(patent_numeric_core)

sort_cols_D = [c for c in ["person_key", "No", "total_score"] if c in D_df.columns]
if sort_cols_D:
    ascending_flags = [True, True, False][:len(sort_cols_D)]
    D_df = D_df.sort_values(by=sort_cols_D, ascending=ascending_flags).reset_index(drop=True)

D_path = OUT_DIR / "D_inventor_candidate_match_detail_v3_2.csv"
D_df.to_csv(D_path, index=False, encoding="utf-8-sig")

# =========================================================
# 7. Table E: No-Level Match Summary Table
# =========================================================
print("Building Table E...")

E_df = all_match_summary_df.copy()

sort_cols_E = [c for c in ["No", "person_key"] if c in E_df.columns]
if sort_cols_E:
    E_df = E_df.sort_values(by=sort_cols_E).reset_index(drop=True)

E_path = OUT_DIR / "E_no_level_match_summary_v3_2.csv"
E_df.to_csv(E_path, index=False, encoding="utf-8-sig")

# =========================================================
# 8. Table F: Inventor-Level Deduplicated Patent Table
#    Core: Compress person_key × pub_no_clean into one row
# =========================================================
print("Building Table F...")

if D_df.empty:
    F_df = pd.DataFrame()
else:
    # Determine sorting: make the first row the 'best record'
    sort_cols = []
    asc = []

    if "person_key" in D_df.columns:
        sort_cols.append("person_key"); asc.append(True)
    if "pub_no_clean" in D_df.columns:
        sort_cols.append("pub_no_clean"); asc.append(True)
    if "total_score" in D_df.columns:
        sort_cols.append("total_score"); asc.append(False)
    if "seed_hit" in D_df.columns:
        sort_cols.append("seed_hit"); asc.append(False)

    D_sorted = D_df.sort_values(by=sort_cols, ascending=asc).reset_index(drop=True)

    grouped_rows = []

    for (person_key, pub_no_clean), sub in D_sorted.groupby(["person_key", "pub_no_clean"], dropna=False):
        sub = sub.copy()

        best_row = sub.iloc[0]

        grouped_rows.append({
            "person_key": person_key,
            "pub_no_clean": pub_no_clean,

            # Representative Identity Information
            "Inductee_example": best_row.get("Inductee", ""),
            "firstname_example": best_row.get("firstname", ""),
            "lastname_example": best_row.get("lastname", ""),

            # Representative Patent Information
            "Publication NO.": best_row.get("Publication NO.", best_row.get("Publication Number", "")),
            "Publication Date": best_row.get("Publication Date", best_row.get("Publication Date", "")),
            "Application NO.": best_row.get("Application NO.", best_row.get("Application Number", "")),
            "Application Date": best_row.get("Application Date", best_row.get("Application Date", "")),
            "Title (English)": best_row.get("Title (English)", best_row.get("Title (English)", "")),
            "Abstract (English)": best_row.get("Abstract (English)", best_row.get("Abstract (English)", "")),
            "Inventor": best_row.get("Inventor", best_row.get("Inventor", "")),
            "First Inventor/Designer": best_row.get("First Inventor/Designer", best_row.get("First Inventor/Designer", "")),
            "Inventor Country/Area": best_row.get("Inventor Country/Area", best_row.get("Inventor Country/Area", "")),
            "Applicant": best_row.get("Applicant", best_row.get("Applicant", "")),
            "Normalized Applicant": best_row.get("Normalized Applicant", best_row.get("Normalized Applicant", "")),
            "IPC Main Class": best_row.get("IPC Main Class", best_row.get("IPC Main Class", "")),
            "IPC Main Subclass": best_row.get("IPC Main Subclass", best_row.get("IPC Main Subclass", "")),
            "IPC Main Group": best_row.get("IPC Main Group", best_row.get("IPC Main Group", "")),
            "IPC Main Subgroup": best_row.get("IPC Main Subgroup", best_row.get("IPC Main Subgroup", "")),
            "CPC": best_row.get("CPC", ""),

            # Best Match Results
            "best_total_score": sub["total_score"].max() if "total_score" in sub.columns else np.nan,
            "best_match_level": best_level_from_series(sub["match_level"]) if "match_level" in sub.columns else "",
            "best_seed_hit": sub["seed_hit"].max() if "seed_hit" in sub.columns else 0,

            # Supporting Information
            "supporting_No_list": safe_join(sorted(sub["No"].dropna().astype(str).unique()), sep="|") if "No" in sub.columns else "",
            "supporting_seed_patent_list": safe_join(sorted(sub["seed_patent_id_clean"].dropna().astype(str).unique()), sep="|") if "seed_patent_id_clean" in sub.columns else "",
            "group_id_list": safe_join(sorted(sub["group_id"].dropna().astype(str).unique()), sep="|") if "group_id" in sub.columns else "",
            "n_supporting_rows": len(sub),

            # Score Distribution
            "n_high_support": int((sub["match_level"] == "high").sum()) if "match_level" in sub.columns else 0,
            "n_medium_support": int((sub["match_level"] == "medium").sum()) if "match_level" in sub.columns else 0,
            "n_low_support": int((sub["match_level"] == "low").sum()) if "match_level" in sub.columns else 0,
        })

    F_df = pd.DataFrame(grouped_rows)

    sort_cols_F = [c for c in ["person_key", "best_total_score"] if c in F_df.columns]
    if sort_cols_F:
        asc_F = [True, False][:len(sort_cols_F)]
        F_df = F_df.sort_values(by=sort_cols_F, ascending=asc_F).reset_index(drop=True)

F_path = OUT_DIR / "F_person_patent_dedup_table_v3_2.csv"
F_df.to_csv(F_path, index=False, encoding="utf-8-sig")

# =========================================================
# 9. Table G: Final Export List Table
# =========================================================
print("Building Table G...")

if F_df.empty:
    G_df = pd.DataFrame()
    G_high_df = pd.DataFrame()
    G_high_medium_df = pd.DataFrame()
else:
    G_df = F_df.copy()

    # Export Flags
    G_df["export_flag_high"] = G_df["best_match_level"].eq("high")
    G_df["export_flag_high_medium"] = G_df["best_match_level"].isin(["high", "medium"])

    # Export Priority
    def export_priority(level):
        level = str(level).strip().lower()
        if level == "high":
            return 1
        if level == "medium":
            return 2
        if level == "low":
            return 3
        return 9

    G_df["export_priority"] = G_df["best_match_level"].apply(export_priority)

    # Keep only columns commonly used for next round export
    preferred_cols = [
        "person_key",
        "Inductee_example",
        "firstname_example",
        "lastname_example",
        "Publication NO.",
        "pub_no_clean",
        "best_match_level",
        "best_total_score",
        "best_seed_hit",
        "supporting_No_list",
        "supporting_seed_patent_list",
        "group_id_list",
        "export_flag_high",
        "export_flag_high_medium",
        "export_priority"
    ]
    preferred_cols = [c for c in preferred_cols if c in G_df.columns]
    G_df = G_df[preferred_cols].copy()

    G_df = G_df.sort_values(
        by=["export_priority", "person_key", "best_total_score"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    G_high_df = G_df[G_df["export_flag_high"]].copy()
    G_high_medium_df = G_df[G_df["export_flag_high_medium"]].copy()

G_path = OUT_DIR / "G_final_export_list_v3_2.csv"
G_df.to_csv(G_path, index=False, encoding="utf-8-sig")

G_high_path = OUT_DIR / "G1_export_high_only_v3_2.csv"
G_high_df.to_csv(G_high_path, index=False, encoding="utf-8-sig")

G_high_medium_path = OUT_DIR / "G2_export_high_medium_v3_2.csv"
G_high_medium_df.to_csv(G_high_medium_path, index=False, encoding="utf-8-sig")

# =========================================================
# 10. Save Table Building Log
# =========================================================
log_rows = [
    {"table": "A", "file": A_path.name, "n_rows": len(A_df)},
    {"table": "B", "file": B_path.name, "n_rows": len(B_df)},
    {"table": "C", "file": C_path.name, "n_rows": len(C_df)},
    {"table": "D", "file": D_path.name, "n_rows": len(D_df)},
    {"table": "E", "file": E_path.name, "n_rows": len(E_df)},
    {"table": "F", "file": F_path.name, "n_rows": len(F_df)},
    {"table": "G", "file": G_path.name, "n_rows": len(G_df)},
    {"table": "G1", "file": G_high_path.name, "n_rows": len(G_high_df)},
    {"table": "G2", "file": G_high_medium_path.name, "n_rows": len(G_high_medium_df)},
]

log_df = pd.DataFrame(log_rows)
log_df.to_csv(OUT_DIR / "build_final_tables_log_v3_2.csv", index=False, encoding="utf-8-sig")

print("Done.")
print(f"Final tables saved to: {OUT_DIR}")