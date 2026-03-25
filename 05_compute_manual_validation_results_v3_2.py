import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================================================
# 0. Path Settings
# =========================================================
BASE = Path("./data")

MANUAL_DIR = BASE / "07_manual_validation_v3_2" / "three_level_stratified_sample"
FILE_REVIEW = MANUAL_DIR / "manual_validation_review_sheet_3levels_v3_2.csv"

OUT_DIR = MANUAL_DIR / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. Parameter Settings
# =========================================================
CONFIDENCE_Z = 1.96   # 95% CI
FIG_DPI = 300

# Main analysis criteria:
# 1) adjudicated_label takes priority
# 2) If adjudicated_label is empty, fall back to manual_label
PRIMARY_LABEL_COL = "final_label"

# =========================================================
# 2. Utility Functions
# =========================================================
def is_missing(x):
    return pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan"

def clean_label(x):
    if is_missing(x):
        return ""
    s = str(x).strip().lower()
    mapping = {
        "correct": "correct",
        "incorrect": "incorrect",
        "uncertain": "uncertain",
        "0": "correct",
        "1": "incorrect",
        "2": "uncertain",
    }
    return mapping.get(s, s)

def wilson_ci(success, n, z=1.96):
    """
    Wilson score interval
    """
    if n == 0:
        return (np.nan, np.nan)

    p = success / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    lower = max(0, center - margin)
    upper = min(1, center + margin)
    return (lower, upper)

def calc_precision_stats(df, label_col):
    """
    precision = correct / (correct + incorrect)
    uncertain does not enter the precision denominator, reported separately
    """
    if label_col not in df.columns:
        return {
            "n_total": len(df),
            "n_correct": np.nan,
            "n_incorrect": np.nan,
            "n_uncertain": np.nan,
            "n_labeled": np.nan,
            "precision": np.nan,
            "precision_ci_lower": np.nan,
            "precision_ci_upper": np.nan,
            "uncertain_rate": np.nan,
            "completion_rate": np.nan,
        }

    labels = df[label_col].apply(clean_label)

    n_total = len(labels)
    n_correct = int((labels == "correct").sum())
    n_incorrect = int((labels == "incorrect").sum())
    n_uncertain = int((labels == "uncertain").sum())
    n_blank = int((labels == "").sum())

    n_labeled = n_correct + n_incorrect
    precision = (n_correct / n_labeled) if n_labeled > 0 else np.nan
    ci_lower, ci_upper = wilson_ci(n_correct, n_labeled, z=CONFIDENCE_Z) if n_labeled > 0 else (np.nan, np.nan)

    uncertain_rate = ((n_uncertain / n_total) if n_total > 0 else np.nan)
    completion_rate = (((n_total - n_blank) / n_total) if n_total > 0 else np.nan)

    return {
        "n_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_uncertain": n_uncertain,
        "n_labeled": n_labeled,
        "precision": precision,
        "precision_ci_lower": ci_lower,
        "precision_ci_upper": ci_upper,
        "uncertain_rate": uncertain_rate,
        "completion_rate": completion_rate,
    }

def simple_agreement_rate(a, b):
    valid = (~a.isna()) & (~b.isna()) & (a != "") & (b != "")
    if valid.sum() == 0:
        return np.nan
    return (a[valid] == b[valid]).mean()

def cohens_kappa(a, b, allowed_labels=("correct", "incorrect", "uncertain")):
    """
    Simple implementation of Cohen's kappa
    """
    a = a.apply(clean_label)
    b = b.apply(clean_label)

    valid = (~a.isna()) & (~b.isna()) & (a != "") & (b != "")
    a = a[valid]
    b = b[valid]

    if len(a) == 0:
        return np.nan

    labels = list(allowed_labels)

    # observed agreement
    po = (a == b).mean()

    # expected agreement
    pa = a.value_counts(normalize=True).reindex(labels, fill_value=0)
    pb = b.value_counts(normalize=True).reindex(labels, fill_value=0)
    pe = (pa * pb).sum()

    if pe == 1:
        return np.nan

    kappa = (po - pe) / (1 - pe)
    return kappa

def add_final_label(df):
    """
    Final label priority:
    adjudicated_label > manual_label > reviewer1_label
    """
    df = df.copy()

    for col in ["adjudicated_label", "manual_label", "reviewer1_label", "reviewer2_label"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_label)

    def choose_label(row):
        for col in ["adjudicated_label", "manual_label", "reviewer1_label"]:
            if col in row.index:
                v = clean_label(row[col])
                if v in ["correct", "incorrect", "uncertain"]:
                    return v
        return ""

    df["final_label"] = df.apply(choose_label, axis=1)
    return df

def safe_value_counts(series):
    return series.apply(clean_label).value_counts(dropna=False)

# =========================================================
# 3. Load Data
# =========================================================
print("Loading manual review sheet...")
df = pd.read_csv(FILE_REVIEW, low_memory=False)

# =========================================================
# 4. Preprocessing
# =========================================================
print("Preparing labels...")

if "sampling_stratum" not in df.columns:
    # Fall back to best_match_level
    if "best_match_level" in df.columns:
        df["sampling_stratum"] = df["best_match_level"].astype(str).str.strip().str.lower()
    else:
        df["sampling_stratum"] = ""

df["sampling_stratum"] = df["sampling_stratum"].astype(str).str.strip().str.lower()

for col in ["manual_label", "reviewer1_label", "reviewer2_label", "adjudicated_label"]:
    if col in df.columns:
        df[col] = df[col].apply(clean_label)

df = add_final_label(df)

# =========================================================
# 5. Original Label Distribution Check
# =========================================================
print("\n=== Label distribution check ===")

for col in ["manual_label", "reviewer1_label", "reviewer2_label", "adjudicated_label", "final_label"]:
    if col in df.columns:
        print(f"\n[{col}]")
        print(safe_value_counts(df[col]))

# =========================================================
# 6. Single Person / Final Label Precision
# =========================================================
print("Computing precision summaries...")

summary_rows = []

# 6.1 final overall
final_stats = calc_precision_stats(df, PRIMARY_LABEL_COL)
summary_rows.append({
    "scope": "overall",
    "label_source": PRIMARY_LABEL_COL,
    **final_stats
})

# 6.2 each stratum
for level in ["high", "medium", "low"]:
    sub = df[df["sampling_stratum"] == level].copy()
    stats = calc_precision_stats(sub, PRIMARY_LABEL_COL)
    summary_rows.append({
        "scope": level,
        "label_source": PRIMARY_LABEL_COL,
        **stats
    })

# 6.3 high + medium combined
sub_hm = df[df["sampling_stratum"].isin(["high", "medium"])].copy()
hm_stats = calc_precision_stats(sub_hm, PRIMARY_LABEL_COL)
summary_rows.append({
    "scope": "high_medium",
    "label_source": PRIMARY_LABEL_COL,
    **hm_stats
})

summary_df = pd.DataFrame(summary_rows)

# =========================================================
# 7. Double Reviewer Agreement Rate
# =========================================================
print("Computing reviewer agreement...")

agreement_rows = []

if "reviewer1_label" in df.columns and "reviewer2_label" in df.columns:
    # overall
    agr = simple_agreement_rate(df["reviewer1_label"], df["reviewer2_label"])
    kap = cohens_kappa(df["reviewer1_label"], df["reviewer2_label"])
    n_valid = (
        (df["reviewer1_label"].astype(str).str.strip() != "") &
        (df["reviewer2_label"].astype(str).str.strip() != "")
    ).sum()

    agreement_rows.append({
        "scope": "overall",
        "n_double_labeled": int(n_valid),
        "agreement_rate": agr,
        "cohens_kappa": kap
    })

    for level in ["high", "medium", "low"]:
        sub = df[df["sampling_stratum"] == level].copy()
        agr = simple_agreement_rate(sub["reviewer1_label"], sub["reviewer2_label"])
        kap = cohens_kappa(sub["reviewer1_label"], sub["reviewer2_label"])
        n_valid = (
            (sub["reviewer1_label"].astype(str).str.strip() != "") &
            (sub["reviewer2_label"].astype(str).str.strip() != "")
        ).sum()

        agreement_rows.append({
            "scope": level,
            "n_double_labeled": int(n_valid),
            "agreement_rate": agr,
            "cohens_kappa": kap
        })

agreement_df = pd.DataFrame(agreement_rows)

# =========================================================
# 8. Generate Record-Level Result Table
# =========================================================
print("Building record-level result table...")

record_df = df.copy()

# Numerical labels for verification
label_num_map = {
    "correct": 0,
    "incorrect": 1,
    "uncertain": 2,
    "": np.nan
}

for col in ["manual_label", "reviewer1_label", "reviewer2_label", "adjudicated_label", "final_label"]:
    if col in record_df.columns:
        record_df[f"{col}_num"] = record_df[col].map(label_num_map)

# Mark if included in precision denominator
record_df["included_in_precision"] = record_df["final_label"].isin(["correct", "incorrect"])
record_df["is_correct"] = record_df["final_label"].eq("correct")
record_df["is_incorrect"] = record_df["final_label"].eq("incorrect")
record_df["is_uncertain"] = record_df["final_label"].eq("uncertain")

# =========================================================
# 9. Save Summary Tables
# =========================================================
print("Saving CSV outputs...")

summary_path = OUT_DIR / "01_manual_validation_precision_summary.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

agreement_path = OUT_DIR / "02_manual_validation_reviewer_agreement.csv"
agreement_df.to_csv(agreement_path, index=False, encoding="utf-8-sig")

record_path = OUT_DIR / "03_manual_validation_record_level_results.csv"
record_df.to_csv(record_path, index=False, encoding="utf-8-sig")

# Additional export: only keep samples needing review
uncertain_df = record_df[record_df["final_label"] == "uncertain"].copy()
uncertain_path = OUT_DIR / "04_manual_validation_uncertain_cases.csv"
uncertain_df.to_csv(uncertain_path, index=False, encoding="utf-8-sig")

disagreement_df = pd.DataFrame()
if "reviewer1_label" in df.columns and "reviewer2_label" in df.columns:
    disagreement_df = record_df[
        (record_df["reviewer1_label"].astype(str).str.strip() != "") &
        (record_df["reviewer2_label"].astype(str).str.strip() != "") &
        (record_df["reviewer1_label"] != record_df["reviewer2_label"])
    ].copy()

disagreement_path = OUT_DIR / "05_manual_validation_disagreement_cases.csv"
disagreement_df.to_csv(disagreement_path, index=False, encoding="utf-8-sig")

# =========================================================
# 10. Output Brief Run Log
# =========================================================
log_rows = [
    {"item": "input_file", "value": str(FILE_REVIEW)},
    {"item": "n_rows_input", "value": len(df)},
    {"item": "primary_label_col", "value": PRIMARY_LABEL_COL},
]

for level in ["high", "medium", "low"]:
    log_rows.append({
        "item": f"n_{level}",
        "value": int((df["sampling_stratum"] == level).sum())
    })

run_log_df = pd.DataFrame(log_rows)
run_log_path = OUT_DIR / "06_manual_validation_run_log.csv"
run_log_df.to_csv(run_log_path, index=False, encoding="utf-8-sig")

# =========================================================
# 11. Visualization
# =========================================================
print("Creating figures...")

# Fig 1: Precision by scope
plot_df = summary_df[summary_df["scope"].isin(["high", "medium", "low", "high_medium", "overall"])].copy()

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(plot_df["scope"], plot_df["precision"])
ax.set_ylim(0, 1)
ax.set_xlabel("Scope")
ax.set_ylabel("Precision")
ax.set_title("Manual validation precision by scope")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_01_precision_by_scope.png", dpi=FIG_DPI)
plt.close()

# Fig 2: Uncertain rate by scope
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(plot_df["scope"], plot_df["uncertain_rate"])
ax.set_ylim(0, 1)
ax.set_xlabel("Scope")
ax.set_ylabel("Uncertain rate")
ax.set_title("Manual validation uncertain rate by scope")
plt.tight_layout()
plt.savefig(OUT_DIR / "fig_02_uncertain_rate_by_scope.png", dpi=FIG_DPI)
plt.close()

# Fig 3: Double reviewer agreement (if available)
if not agreement_df.empty:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(agreement_df["scope"], agreement_df["agreement_rate"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Scope")
    ax.set_ylabel("Agreement rate")
    ax.set_title("Reviewer agreement by scope")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_03_reviewer_agreement.png", dpi=FIG_DPI)
    plt.close()

# =========================================================
# 12. Completion
# =========================================================
print("Done.")
print(f"Results saved to: {OUT_DIR}")
print(f"Precision summary: {summary_path}")
print(f"Agreement summary: {agreement_path}")
print(f"Record-level results: {record_path}")