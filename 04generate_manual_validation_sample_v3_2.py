import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 0. Path Settings
# =========================================================
BASE = Path("./data")

FINAL_DIR = BASE / "05_final_tables_v3_2"
FILE_F = FINAL_DIR / "F_person_patent_dedup_table_v3_2.csv"

OUT_DIR = BASE / "07_manual_validation_v3_2" / "three_level_stratified_sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. Parameter Settings
# =========================================================
RANDOM_SEED = 202603

# Fixed sample size per level
SAMPLE_SIZE_BY_LEVEL = {
    "high": 100,
    "medium": 100,
    "low": 100
}

# Whether to generate blind review sheet (hide stratification information)
GENERATE_BLIND_REVIEW_SHEET = True

# =========================================================
# 2. Reading Data
# =========================================================
print("Loading F table...")
df = pd.read_csv(FILE_F)

# =========================================================
# 3. Data Preprocessing
# =========================================================
print("Preparing sampling frame...")

df["best_match_level"] = df["best_match_level"].astype(str).str.strip().str.lower()

# Only keep high / medium / low
df = df[df["best_match_level"].isin(["high", "medium", "low"])].copy()
df = df.reset_index(drop=True)

print("\n=== best_match_level distribution in F ===")
print(df["best_match_level"].value_counts(dropna=False))

# =========================================================
# 4. Stratified Sampling
# =========================================================
np.random.seed(RANDOM_SEED)

samples = []
sampling_log_rows = []

for level, n_target in SAMPLE_SIZE_BY_LEVEL.items():
    sub = df[df["best_match_level"] == level].copy()
    n_available = len(sub)
    n_sample = min(n_target, n_available)

    if n_available == 0:
        print(f"Warning: no records available for level = {level}")
        sampled = sub.copy()
    else:
        sampled = sub.sample(n=n_sample, replace=False, random_state=RANDOM_SEED)

    sampled["sampling_stratum"] = level
    samples.append(sampled)

    sampling_log_rows.append({
        "level": level,
        "n_available": n_available,
        "n_target": n_target,
        "n_sampled": n_sample
    })

sample_df = pd.concat(samples, axis=0).reset_index(drop=True)

# Shuffle the overall order to avoid seeing the same level consecutively during manual review
sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# =========================================================
# 5. Add Sample ID and Manual Review Columns
# =========================================================
sample_df["sample_id"] = range(1, len(sample_df) + 1)

# Single reviewer fields
sample_df["manual_label"] = ""
sample_df["manual_reason"] = ""
sample_df["reviewer"] = ""
sample_df["review_date"] = ""

# Double reviewer fields (recommended)
sample_df["reviewer1_label"] = ""
sample_df["reviewer1_reason"] = ""
sample_df["reviewer1_name"] = ""
sample_df["reviewer1_date"] = ""

sample_df["reviewer2_label"] = ""
sample_df["reviewer2_reason"] = ""
sample_df["reviewer2_name"] = ""
sample_df["reviewer2_date"] = ""

# If double review is adopted, the final adjudication can be filled in later
sample_df["adjudicated_label"] = ""
sample_df["adjudicated_reason"] = ""
sample_df["adjudicator"] = ""
sample_df["adjudication_date"] = ""

# =========================================================
# 6. Output Manual Review Sheet (Non-Blind)
# =========================================================
review_cols = [
    "sample_id",
    "sampling_stratum",
    "person_key",
    "Inductee_example",
    "firstname_example",
    "lastname_example",
    "Publication NO.",
    "pub_no_clean",
    "Publication Date",
    "Application Date",
    "Title (English)",
    "Applicant",
    "Normalized Applicant",
    "Inventor",
    "Inventor Country/Area",
    "IPC Main Class",
    "IPC Main Subclass",
    "IPC Main Group",
    "IPC Main Subgroup",
    "CPC",
    "best_match_level",
    "best_total_score",
    "best_seed_hit",
    "supporting_No_list",
    "supporting_seed_patent_list",
    "group_id_list",
    "n_supporting_rows",
    "n_high_support",
    "n_medium_support",
    "n_low_support",
    "manual_label",
    "manual_reason",
    "reviewer",
    "review_date",
    "reviewer1_label",
    "reviewer1_reason",
    "reviewer1_name",
    "reviewer1_date",
    "reviewer2_label",
    "reviewer2_reason",
    "reviewer2_name",
    "reviewer2_date",
    "adjudicated_label",
    "adjudicated_reason",
    "adjudicator",
    "adjudication_date"
]

review_cols = [c for c in review_cols if c in sample_df.columns]
review_df = sample_df[review_cols].copy()

review_path = OUT_DIR / "manual_validation_review_sheet_3levels_v3_2.csv"
review_df.to_csv(review_path, index=False, encoding="utf-8-sig")

# =========================================================
# 7. Output Blind Review Sheet (Recommended)
#    Hide best_match_level / score / stratum to reduce bias
# =========================================================
if GENERATE_BLIND_REVIEW_SHEET:
    blind_drop_cols = {
        "sampling_stratum",
        "best_match_level",
        "best_total_score",
        "best_seed_hit",
        "n_high_support",
        "n_medium_support",
        "n_low_support"
    }

    blind_cols = [c for c in review_cols if c not in blind_drop_cols]
    blind_df = sample_df[blind_cols].copy()

    blind_path = OUT_DIR / "manual_validation_blind_review_sheet_3levels_v3_2.csv"
    blind_df.to_csv(blind_path, index=False, encoding="utf-8-sig")

# =========================================================
# 8. Output Complete Sample Information Table (Read-Only Archive)
# =========================================================
fullinfo_path = OUT_DIR / "manual_validation_sample_fullinfo_3levels_v3_2.csv"
sample_df.to_csv(fullinfo_path, index=False, encoding="utf-8-sig")

# =========================================================
# 9. Output Sampling Log
# =========================================================
sampling_log_df = pd.DataFrame(sampling_log_rows)
sampling_log_df["random_seed"] = RANDOM_SEED

sampling_log_path = OUT_DIR / "manual_validation_sampling_log_3levels_v3_2.csv"
sampling_log_df.to_csv(sampling_log_path, index=False, encoding="utf-8-sig")

# =========================================================
# 10. Output Brief Summary
# =========================================================
summary_rows = [
    {"metric": "random_seed", "value": RANDOM_SEED},
    {"metric": "total_sampled", "value": len(sample_df)},
    {"metric": "n_high_sampled", "value": int((sample_df["sampling_stratum"] == "high").sum())},
    {"metric": "n_medium_sampled", "value": int((sample_df["sampling_stratum"] == "medium").sum())},
    {"metric": "n_low_sampled", "value": int((sample_df["sampling_stratum"] == "low").sum())},
]
summary_df = pd.DataFrame(summary_rows)

summary_path = OUT_DIR / "manual_validation_sampling_summary_3levels_v3_2.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

# =========================================================
# 11. Completion
# =========================================================
print("\nManual validation sample generated successfully.")
print(f"Review sheet: {review_path}")
if GENERATE_BLIND_REVIEW_SHEET:
    print(f"Blind review sheet: {blind_path}")
print(f"Full info sheet: {fullinfo_path}")
print(f"Sampling log: {sampling_log_path}")
print(f"Summary: {summary_path}")