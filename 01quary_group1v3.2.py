import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================================================
# 0. Parameter Settings
# =========================================================
GROUP_ID = 1
CURRENT_YEAR = 2026

BASE = Path("./data")

FILE_MATCHED = BASE / "NIHF_patent_matched.csv"
FILE_MAPPING = BASE / "inventor_query_mapping_with_check.csv"
FILE_SEED_RAW = BASE / "NIHF_raw" / "NIHF_raw_download.xlsx"
FILE_GROUP = BASE / "NIHF_all" / f"{GROUP_ID}.xlsx"

OUT_DIR = BASE / f"group{GROUP_ID}_match_output_v3_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. Basic Utility Functions
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

def split_multi_value(x):
    """
    Suitable for splitting:
    Inventors / Standardized Assignees / Inventor Countries/Regions / CPC
    """
    if is_missing(x):
        return []
    s = str(x)
    parts = re.split(r"[;；,，|\n]+", s)
    parts = [clean_text(p) for p in parts if clean_text(p)]
    return parts

def unique_keep_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_spaces_and_dot(s):
    s = clean_text(s)
    s = s.replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_name(name):
    name = normalize_spaces_and_dot(name)
    return [t for t in name.split() if t]

def token_initial(token):
    return token[0] if token else ""

def normalize_org(s):
    s = clean_text(s)
    s = re.sub(r"\s+", " ", s)
    return s

# =========================================================
# 2. person_key
# =========================================================
def build_person_key(firstname, lastname, born_year):
    fn = normalize_spaces_and_dot(firstname)
    ln = normalize_spaces_and_dot(lastname)
    by = extract_year_from_date(born_year)

    if by is not None:
        return f"{fn}||{ln}||{by}"
    return f"{fn}||{ln}"

# =========================================================
# 3. Name Equivalence (Nicknames / Full Names) Bidirectional Mapping
# =========================================================
FIRSTNAME_GROUPS = [
    ["don", "donald"],
    ["bob", "robert", "rob"],
    ["jim", "james"],
    ["bill", "will", "william"],
    ["tom", "thomas"],
    ["jack", "john"],
    ["dick", "richard", "rick"],
    ["mike", "michael"],
    ["dave", "david"],
    ["ed", "edward", "ted"],
    ["ben", "benjamin"],
    ["dan", "daniel"],
    ["sam", "samuel"],
    ["joe", "joseph"],
    ["frank", "francis", "franklin"],
    ["hank", "henry", "harry"],
    ["alex", "alexander"],
    ["andy", "andrew"],
    ["tony", "anthony"],
    ["pat", "patrick"],
    ["ron", "ronald"],
    ["steve", "steven", "stephen"],
    ["jeff", "jeffrey"],
    ["greg", "gregory"],
    ["ken", "kenneth"],
    ["matt", "matthew"],
    ["chris", "christopher"],
    ["larry", "lawrence"],
    ["walt", "walter"],
    ["chuck", "charles"],
]

FIRSTNAME_EQUIV = {}
for grp in FIRSTNAME_GROUPS:
    s = set(grp)
    for g in grp:
        FIRSTNAME_EQUIV[g] = s

def get_firstname_family(firstname):
    toks = tokenize_name(firstname)
    if not toks:
        return set()
    first_main = toks[0]
    fam = set([first_main])
    fam.update(FIRSTNAME_EQUIV.get(first_main, {first_main}))
    return fam

def find_first_token_match(tokens, first_family):
    if not tokens:
        return None, None

    initials = set([token_initial(x) for x in first_family if x])

    for t in tokens:
        if t in first_family:
            return t, "family_full"

    for t in tokens:
        if len(t) == 1 and t in initials:
            return t, "initial"

    return None, None

# =========================================================
# 4. Name Profile Construction
# =========================================================
def build_name_profile(raw_name, target_firstname, target_lastname, source="seed_raw", strength=None):
    raw_name_clean = normalize_spaces_and_dot(raw_name)
    if raw_name_clean == "":
        return None

    target_last = normalize_spaces_and_dot(target_lastname)
    first_family = get_firstname_family(target_firstname)

    if not first_family or target_last == "":
        return None

    if strength is None:
        strength = "strong" if source == "seed_raw" else "weak"

    tokens = tokenize_name(raw_name_clean)
    if not tokens:
        return None

    if target_last not in tokens:
        return None

    rest = []
    removed_last = False
    for t in tokens:
        if (not removed_last) and t == target_last:
            removed_last = True
            continue
        rest.append(t)

    matched_first_token, first_match_mode = find_first_token_match(rest, first_family)

    if matched_first_token is None:
        return None

    filtered = []
    removed_first = False
    for t in rest:
        if (not removed_first) and t == matched_first_token:
            removed_first = True
            continue
        filtered.append(t)

    middle_tokens = filtered
    middle_initials = [token_initial(t) for t in middle_tokens if t]

    return {
        "raw_name": raw_name_clean,
        "source": source,
        "strength": strength,
        "target_first_family": sorted(first_family),
        "target_last": target_last,
        "first_match_mode": first_match_mode,
        "matched_first_token": matched_first_token,
        "middle_tokens": middle_tokens,
        "middle_initials": middle_initials,
        "tokens": tokens,
    }

def build_fallback_profile_from_inductee(inductee, firstname, lastname):
    return build_name_profile(
        raw_name=inductee,
        target_firstname=firstname,
        target_lastname=lastname,
        source="inductee_fallback",
        strength="weak"
    )

# =========================================================
# 5. Name Comparison
# =========================================================
def middle_consistency(seed_mids, cand_mids):
    seed_mids = [m for m in seed_mids if m]
    cand_mids = [m for m in cand_mids if m]

    if not seed_mids and not cand_mids:
        return "exact"
    if seed_mids and not cand_mids:
        return "missing"
    if not seed_mids and cand_mids:
        return "seed_none_candidate_extra"

    seed_set = set(seed_mids)
    cand_set = set(cand_mids)

    if seed_mids == cand_mids:
        return "exact"
    if cand_set.issubset(seed_set) or seed_set.issubset(cand_set):
        return "compatible"

    return "conflict"

def compare_candidate_name_to_seed_profile(candidate_name, seed_profile):
    seed_firstname_proxy = seed_profile["target_first_family"][0] if seed_profile["target_first_family"] else ""
    cand_profile = build_name_profile(
        raw_name=candidate_name,
        target_firstname=seed_firstname_proxy,
        target_lastname=seed_profile["target_last"],
        source="candidate",
        strength="candidate"
    )
    if cand_profile is None:
        return 0, "not_target"

    mid_status = middle_consistency(seed_profile["middle_initials"], cand_profile["middle_initials"])
    seed_strength = seed_profile.get("strength", "strong")
    cand_first_mode = cand_profile["first_match_mode"]

    if seed_strength == "strong" and mid_status == "conflict":
        return 0, "middle_conflict_strong"

    if cand_first_mode == "family_full":
        if mid_status == "exact":
            return (50 if seed_strength == "strong" else 42), "full_family_exact"
        elif mid_status == "compatible":
            return (44 if seed_strength == "strong" else 36), "full_family_compatible"
        elif mid_status == "missing":
            return (34 if seed_strength == "strong" else 30), "full_family_missing_middle"
        elif mid_status == "seed_none_candidate_extra":
            return (28 if seed_strength == "strong" else 30), "full_family_candidate_extra_middle"
        elif mid_status == "conflict":
            return 12, "full_family_weak_middle_conflict"

    if cand_first_mode == "initial":
        if mid_status == "exact":
            return 36, "initial_exact"
        elif mid_status == "compatible":
            return 30, "initial_compatible"
        elif mid_status == "missing":
            return 20, "initial_missing_middle"
        elif mid_status == "seed_none_candidate_extra":
            return 15, "initial_candidate_extra_middle"
        elif mid_status == "conflict":
            return 6, "initial_weak_middle_conflict"

    return 0, "unmatched"

def best_name_match(candidate_inventor_field, seed_profiles, firstname, lastname, inductee):
    candidate_names = split_multi_value(candidate_inventor_field)
    best = {
        "name_score": 0,
        "name_rule": "none",
        "matched_candidate_name": "",
        "matched_seed_name": "",
        "matched_seed_source": ""
    }

    if not seed_profiles:
        fb = build_fallback_profile_from_inductee(inductee, firstname, lastname)
        if fb is not None:
            seed_profiles = [fb]

    for cand_name in candidate_names:
        for seed_profile in seed_profiles:
            score, rule = compare_candidate_name_to_seed_profile(cand_name, seed_profile)
            if score > best["name_score"]:
                best = {
                    "name_score": score,
                    "name_rule": rule,
                    "matched_candidate_name": cand_name,
                    "matched_seed_name": seed_profile["raw_name"],
                    "matched_seed_source": seed_profile["source"]
                }

    return best

# =========================================================
# 6. Assignee Intersection Functions (For Pool Fallback + Formal Scoring)
# =========================================================
def assignee_overlap_score(candidate_std_assignee, seed_assignees):
    cand = [normalize_org(x) for x in split_multi_value(candidate_std_assignee)]
    seed = [normalize_org(x) for x in seed_assignees if normalize_org(x)]

    cand_set = set(cand)
    seed_set = set(seed)

    if not cand_set or not seed_set:
        return 0, "no_assignee_info"

    exact_inter = cand_set & seed_set
    if exact_inter:
        return 22, "assignee_exact_overlap"

    contain_hits = []
    for c in cand_set:
        for s in seed_set:
            if c in s or s in c:
                contain_hits.append((c, s))
    if contain_hits:
        return 14, "assignee_containment"

    return 0, "assignee_no_overlap"

def assignee_pool_overlap(candidate_std_assignee, seed_assignees):
    """
    For candidate pool fallback, more lenient than formal scoring:
    Return True as long as there is intersection or containment relationship
    """
    score, rule = assignee_overlap_score(candidate_std_assignee, seed_assignees)
    return score > 0, rule

# =========================================================
# 7. Co-Inventors / Country/Region
# =========================================================
def person_signature(name):
    toks = tokenize_name(name)
    if len(toks) == 0:
        return ""

    if len(toks) == 1:
        return toks[0]

    sig1 = f"{toks[0]}|{token_initial(toks[1])}"
    sig2 = f"{toks[-1]}|{token_initial(toks[0])}"
    return f"{sig1}||{sig2}"

def explode_person_signature(sig):
    if not sig:
        return set()
    return set([x for x in sig.split("||") if x])

def build_coinventor_signature_set(inventor_field, target_matched_name=""):
    names = split_multi_value(inventor_field)
    target_clean = normalize_spaces_and_dot(target_matched_name)

    sigs = set()
    for n in names:
        n_clean = normalize_spaces_and_dot(n)
        if target_clean and n_clean == target_clean:
            continue
        sig = person_signature(n_clean)
        sigs.update(explode_person_signature(sig))
    return set([x for x in sigs if x])

def coinventor_overlap_score(candidate_inventor_field, seed_coinventor_sigs, matched_candidate_name=""):
    cand_sigs = build_coinventor_signature_set(candidate_inventor_field, matched_candidate_name)
    seed_sigs = set(seed_coinventor_sigs) if seed_coinventor_sigs else set()

    if not cand_sigs or not seed_sigs:
        return 0, 0, "no_coinventor_info"

    inter = cand_sigs & seed_sigs
    n = len(inter)

    if n >= 3:
        return 18, n, "coinventor_overlap_3plus"
    elif n == 2:
        return 14, n, "coinventor_overlap_2"
    elif n == 1:
        return 8, n, "coinventor_overlap_1"
    else:
        return 0, 0, "coinventor_no_overlap"

def country_overlap_score(candidate_country_field, seed_country_set):
    cand = set([clean_text(x) for x in split_multi_value(candidate_country_field) if clean_text(x)])
    seed = set([clean_text(x) for x in seed_country_set if clean_text(x)])

    if not cand or not seed:
        return 0, "no_country_info"

    if cand & seed:
        return 8, "country_overlap"

    return -8, "country_conflict"

# =========================================================
# 8. IPC / CPC
# =========================================================
def extract_alnum_prefix(s):
    return re.sub(r"[^A-Z0-9/]", "", str(s).upper())

def expand_single_cpc(code):
    code = extract_alnum_prefix(code)
    if code == "":
        return []

    if "/" in code:
        left, right = code.split("/", 1)
    else:
        left, right = code, None

    out = []

    m1 = re.match(r"^([A-Z]\d{2})", left)
    if m1:
        out.append(m1.group(1))

    m2 = re.match(r"^([A-Z]\d{2}[A-Z])", left)
    if m2:
        out.append(m2.group(1))

    m3 = re.match(r"^([A-Z]\d{2}[A-Z]\d+)", left)
    if m3:
        out.append(m3.group(1))

    out.append(code)
    return unique_keep_order(out)

def expand_cpc_field(cpc_field):
    parts = split_multi_value(cpc_field)
    expanded = []
    for p in parts:
        expanded.extend(expand_single_cpc(p))
    return set(unique_keep_order(expanded))

def cpc_token_level(token):
    token = extract_alnum_prefix(token)
    if "/" in token:
        return 4
    if re.match(r"^[A-Z]\d{2}[A-Z]\d+$", token):
        return 3
    if re.match(r"^[A-Z]\d{2}[A-Z]$", token):
        return 2
    if re.match(r"^[A-Z]\d{2}$", token):
        return 1
    return 0

def cpc_overlap_score(candidate_cpc_field, seed_cpc_expanded_set):
    cand_set = expand_cpc_field(candidate_cpc_field)
    seed_set = set(seed_cpc_expanded_set) if seed_cpc_expanded_set else set()

    if not cand_set or not seed_set:
        return 0, "", "no_cpc_info"

    inter = cand_set & seed_set
    if not inter:
        return 0, "", "cpc_no_overlap"

    best_token = sorted(list(inter), key=lambda x: (cpc_token_level(x), len(x)), reverse=True)[0]
    level = cpc_token_level(best_token)

    if level == 4:
        return 24, best_token, "cpc_full_group_overlap"
    elif level == 3:
        return 18, best_token, "cpc_main_group_overlap"
    elif level == 2:
        return 12, best_token, "cpc_subclass_overlap"
    elif level == 1:
        return 6, best_token, "cpc_class_overlap"
    return 0, "", "cpc_unknown"

def ipc_overlap_score(candidate_big, candidate_small, seed_big_set, seed_small_set):
    c_big = clean_text(candidate_big)
    c_small = clean_text(candidate_small)

    seed_big = set([clean_text(x) for x in seed_big_set if clean_text(x)])
    seed_small = set([clean_text(x) for x in seed_small_set if clean_text(x)])

    if c_small and c_small in seed_small:
        return 12, "ipc_small_overlap"
    if c_big and c_big in seed_big:
        return 7, "ipc_big_overlap"
    return 0, "ipc_no_overlap"

def class_similarity_score(candidate_big, candidate_small, candidate_cpc,
                           seed_big_set, seed_small_set, seed_cpc_expanded_set):
    cpc_score, best_cpc_token, cpc_rule = cpc_overlap_score(candidate_cpc, seed_cpc_expanded_set)
    ipc_score, ipc_rule = ipc_overlap_score(candidate_big, candidate_small, seed_big_set, seed_small_set)

    score = max(cpc_score, ipc_score)
    if cpc_score > 0 and ipc_score > 0:
        score = min(score + 3, 27)

    rule = f"{cpc_rule}|{ipc_rule}"
    return score, best_cpc_token, rule

# =========================================================
# 9. Time Scoring
# =========================================================
def time_score(candidate_year, seed_years, born_year=None, died_year=None):
    if candidate_year is None or pd.isna(candidate_year):
        return 0, "no_candidate_year"

    candidate_year = int(candidate_year)

    if born_year is not None:
        start_y = born_year + 15
        end_y = min(born_year + 90, CURRENT_YEAR) if died_year is None else min(born_year + 90, died_year, CURRENT_YEAR)
        if candidate_year < start_y or candidate_year > end_y:
            return -15, "out_of_life_window"

    seed_years = [int(y) for y in seed_years if pd.notna(y)]
    if seed_years:
        gap = min(abs(candidate_year - y) for y in seed_years)
        if gap <= 5:
            return 10, "year_gap_le_5"
        elif gap <= 15:
            return 8, "year_gap_le_15"
        else:
            return 5, "within_window_far_from_seed"
    return 5, "within_window"

def time_window_ok(candidate_year, born_year=None, died_year=None):
    """
    For candidate pool fallback: only check if within reasonable career window
    """
    if candidate_year is None or pd.isna(candidate_year):
        return False

    candidate_year = int(candidate_year)

    if born_year is None:
        return True

    start_y = born_year + 15
    end_y = min(born_year + 90, CURRENT_YEAR) if died_year is None else min(born_year + 90, died_year, CURRENT_YEAR)

    return start_y <= candidate_year <= end_y

def match_level(total_score):
    if total_score >= 60:
        return "high"
    elif total_score >= 40:
        return "medium"
    else:
        return "low"

# =========================================================
# 10. Person-Level Coarse Backfill Rules (v3.2)
# =========================================================
def coarse_candidate_for_person(pub_no_clean,
                                candidate_inventor_field,
                                candidate_std_assignee,
                                candidate_year,
                                firstname,
                                lastname,
                                seed_profiles,
                                inductee,
                                person_seed_patent_ids,
                                person_seed_assignees,
                                born_year,
                                died_year):
    """
    v3.2 Rules for entering shared candidate pool:

    1. Hit any seed patent id -> Force inclusion
    2. If lastname length >= 4, surname hit is sufficient to enter pool
    3. If lastname length <= 3, need:
       - Surname hit + first family / initial
       - Or best_name_match > 0
    4. New fallback:
       - Standardized assignee has intersection/containment relationship
       - And time is reasonable
       => Allow entry into pool
    """

    # 1) Force inclusion of seed patent
    if pub_no_clean and pub_no_clean in set(person_seed_patent_ids):
        return True, "forced_by_seed_patent"

    lastname_clean = normalize_spaces_and_dot(lastname)
    text = clean_text(candidate_inventor_field)

    # 2) First look at name side
    if lastname_clean and lastname_clean in text:
        # Long surname: surname hit is sufficient
        if len(lastname_clean) >= 4:
            return True, "lastname_only_pool"

        # Short surname: more strict
        first_family = get_firstname_family(firstname)
        initials = set([token_initial(x) for x in first_family if x])

        candidate_names = split_multi_value(candidate_inventor_field)
        for cand_name in candidate_names:
            toks = tokenize_name(cand_name)
            if lastname_clean not in toks:
                continue

            if any(t in first_family for t in toks):
                return True, "short_lastname_plus_first_family"

            if any((len(t) == 1 and t in initials) for t in toks):
                return True, "short_lastname_plus_first_initial"

        nm = best_name_match(candidate_inventor_field, seed_profiles, firstname, lastname, inductee)
        if nm["name_score"] > 0:
            return True, f"short_lastname_best_name_match:{nm['name_rule']}"

    # 3) New: Assignee intersection entry into pool fallback
    assignee_keep, assignee_rule = assignee_pool_overlap(candidate_std_assignee, person_seed_assignees)
    if assignee_keep and time_window_ok(candidate_year, born_year, died_year):
        return True, f"assignee_pool_fallback:{assignee_rule}"

    return False, "not_in_person_pool"

# =========================================================
# 11. Data Reading
# =========================================================
matched_df = pd.read_csv(FILE_MATCHED)
mapping_df = pd.read_csv(FILE_MAPPING)
seed_raw_df = pd.read_excel(FILE_SEED_RAW)
group_df = pd.read_excel(FILE_GROUP)

# =========================================================
# 12. Group Inventors + person_key
# =========================================================
group_inventors = mapping_df[mapping_df["group_id"] == GROUP_ID].copy()
group_inventors["patent_id_clean2"] = group_inventors["patent_id"].apply(patent_numeric_core)
group_inventors["person_key"] = group_inventors.apply(
    lambda r: build_person_key(r.get("firstname", ""), r.get("lastname", ""), r.get("Born_year", "")),
    axis=1
)

base_cols = ["No", "Inductee", "firstname", "lastname", "Tech", "patent_id", "Inducted_year", "Born_year", "Died_year"]
base_cols = [c for c in base_cols if c in matched_df.columns]
group_inventors = group_inventors.merge(
    matched_df[base_cols].drop_duplicates(),
    on=base_cols,
    how="left"
)

# =========================================================
# 13. Organize Seed Raw / Group Patents
# =========================================================
seed_raw_df["Publication Number Clean"] = seed_raw_df["Publication (Announcement) Number"].apply(patent_numeric_core)
seed_raw_df["Application Year"] = seed_raw_df["Application Date"].apply(extract_year_from_date)
seed_raw_df["Publication Year"] = seed_raw_df["Publication (Announcement) Date"].apply(extract_year_from_date)

group_df["Publication Number Clean"] = group_df["Publication (Announcement) Number"].apply(patent_numeric_core)
group_df["Application Year"] = group_df["Application Date"].apply(extract_year_from_date)
group_df["Publication Year"] = group_df["Publication (Announcement) Date"].apply(extract_year_from_date)
group_df["Candidate Year"] = group_df["Application Year"].fillna(group_df["Publication Year"])
group_df["Group ID"] = GROUP_ID

# =========================================================
# 14. No-Level Seed Features
# =========================================================
seed_feature_rows = []

for _, inv in group_inventors.iterrows():
    no = inv["No"]
    person_key = inv["person_key"]
    inductee = inv["Inductee"]
    firstname = inv["firstname"]
    lastname = inv["lastname"]
    patent_id_clean = patent_numeric_core(inv["patent_id"])

    seed_patents = seed_raw_df[seed_raw_df["Publication Number Clean"] == patent_id_clean].copy()

    seed_profiles = []
    seed_assignees = set()
    seed_ipc_big = set()
    seed_ipc_small = set()
    seed_cpc_expanded = set()
    seed_years = []
    seed_country = set()
    seed_coinventor_sigs = set()
    seed_inventor_field_raw = []

    for _, r in seed_patents.iterrows():
        inv_field = r.get("Inventors", "")
        seed_inventor_field_raw.append(inv_field)

        inventor_names = split_multi_value(inv_field)

        row_target_profiles = []
        for raw_name in inventor_names:
            prof = build_name_profile(raw_name, firstname, lastname, source="seed_raw", strength="strong")
            if prof is not None:
                row_target_profiles.append(prof)

        seed_profiles.extend(row_target_profiles)

        matched_target_raw_names = [p["raw_name"] for p in row_target_profiles]
        for raw_name in inventor_names:
            raw_name_norm = normalize_spaces_and_dot(raw_name)
            if raw_name_norm in matched_target_raw_names:
                continue
            sig = person_signature(raw_name_norm)
            seed_coinventor_sigs.update(explode_person_signature(sig))

        for a in split_multi_value(r.get("Standardized Assignees", "")):
            seed_assignees.add(normalize_org(a))

        for c in split_multi_value(r.get("Inventor Countries/Regions", "")):
            seed_country.add(clean_text(c))

        big = clean_text(r.get("IPC Main Classification - Major Class", ""))
        small = clean_text(r.get("IPC Main Classification - Subclass", ""))
        if big:
            seed_ipc_big.add(big)
        if small:
            seed_ipc_small.add(small)

        seed_cpc_expanded.update(expand_cpc_field(r.get("CPC", "")))

        if pd.notna(r.get("Application Year", None)):
            seed_years.append(int(r["Application Year"]))
        if pd.notna(r.get("Publication Year", None)):
            seed_years.append(int(r["Publication Year"]))

    if not seed_profiles:
        fb = build_fallback_profile_from_inductee(inductee, firstname, lastname)
        if fb is not None:
            seed_profiles = [fb]

    dedup_seed_profiles = []
    seen_seed_names = set()
    for p in seed_profiles:
        key = (p["raw_name"], p["source"], p["strength"])
        if key not in seen_seed_names:
            seen_seed_names.add(key)
            dedup_seed_profiles.append(p)
    seed_profiles = dedup_seed_profiles

    seed_feature_rows.append({
        "No": no,
        "person_key": person_key,
        "seed_patent_id_clean": patent_id_clean,
        "seed_patent_found": len(seed_patents) > 0,
        "seed_inventor_field_raw": " | ".join(unique_keep_order(seed_inventor_field_raw)),
        "seed_target_names": " | ".join([p["raw_name"] for p in seed_profiles]),
        "seed_target_name_sources": " | ".join([f"{p['source']}:{p['strength']}" for p in seed_profiles]),
        "seed_assignees": " | ".join(sorted(seed_assignees)),
        "seed_ipc_big": " | ".join(sorted(seed_ipc_big)),
        "seed_ipc_small": " | ".join(sorted(seed_ipc_small)),
        "seed_cpc_expanded": " | ".join(sorted(seed_cpc_expanded)),
        "seed_country": " | ".join(sorted(seed_country)),
        "seed_years": " | ".join([str(y) for y in sorted(set(seed_years))]),
        "seed_coinventor_sigs": " | ".join(sorted(seed_coinventor_sigs)),
        "_seed_profiles_obj": seed_profiles,
        "_seed_assignees_obj": sorted(seed_assignees),
        "_seed_ipc_big_obj": sorted(seed_ipc_big),
        "_seed_ipc_small_obj": sorted(seed_ipc_small),
        "_seed_cpc_expanded_obj": sorted(seed_cpc_expanded),
        "_seed_country_obj": sorted(seed_country),
        "_seed_years_obj": sorted(set(seed_years)),
        "_seed_coinventor_sigs_obj": sorted(seed_coinventor_sigs),
    })

seed_features_df = pd.DataFrame(seed_feature_rows)
group_inventors = group_inventors.merge(seed_features_df, on=["No", "person_key"], how="left")

# =========================================================
# 15. Person-Level Feature Aggregation
# =========================================================
person_feature_rows = []

for person_key, sub in group_inventors.groupby("person_key", dropna=False):
    sub = sub.copy()

    firstname = sub["firstname"].iloc[0]
    lastname = sub["lastname"].iloc[0]
    inductee = sub["Inductee"].iloc[0]
    born_year = extract_year_from_date(sub["Born_year"].iloc[0])
    died_year = extract_year_from_date(sub["Died_year"].iloc[0])

    merged_profiles = []
    merged_assignees = set()
    merged_ipc_big = set()
    merged_ipc_small = set()
    merged_cpc_expanded = set()
    merged_country = set()
    merged_years = set()
    merged_coinventor_sigs = set()
    merged_seed_patent_ids = set()
    merged_seed_names = []

    seed_no_list = sub["No"].tolist()

    for _, r in sub.iterrows():
        profs = r.get("_seed_profiles_obj", [])
        if isinstance(profs, list):
            merged_profiles.extend(profs)

        spid = r.get("seed_patent_id_clean", "")
        if spid:
            merged_seed_patent_ids.add(spid)

        stn = r.get("seed_target_names", "")
        if not is_missing(stn):
            merged_seed_names.extend([x.strip() for x in str(stn).split("|") if x.strip()])

        vals = r.get("_seed_assignees_obj", [])
        if isinstance(vals, list):
            merged_assignees.update(vals)

        vals = r.get("_seed_ipc_big_obj", [])
        if isinstance(vals, list):
            merged_ipc_big.update(vals)

        vals = r.get("_seed_ipc_small_obj", [])
        if isinstance(vals, list):
            merged_ipc_small.update(vals)

        vals = r.get("_seed_cpc_expanded_obj", [])
        if isinstance(vals, list):
            merged_cpc_expanded.update(vals)

        vals = r.get("_seed_country_obj", [])
        if isinstance(vals, list):
            merged_country.update(vals)

        vals = r.get("_seed_years_obj", [])
        if isinstance(vals, list):
            merged_years.update(vals)

        vals = r.get("_seed_coinventor_sigs_obj", [])
        if isinstance(vals, list):
            merged_coinventor_sigs.update(vals)

    dedup_profiles = []
    seen_profile_keys = set()
    for p in merged_profiles:
        k = (p["raw_name"], p["source"], p["strength"])
        if k not in seen_profile_keys:
            seen_profile_keys.add(k)
            dedup_profiles.append(p)

    if not dedup_profiles:
        fb = build_fallback_profile_from_inductee(inductee, firstname, lastname)
        if fb is not None:
            dedup_profiles = [fb]

    person_feature_rows.append({
        "person_key": person_key,
        "firstname": firstname,
        "lastname": lastname,
        "Inductee_example": inductee,
        "Born_year_example": sub["Born_year"].iloc[0],
        "Died_year_example": sub["Died_year"].iloc[0],
        "seed_no_list": "|".join(map(str, seed_no_list)),
        "person_seed_patent_ids": "|".join(sorted(merged_seed_patent_ids)),
        "person_seed_names": " | ".join(unique_keep_order(merged_seed_names)),
        "person_seed_assignees": " | ".join(sorted(merged_assignees)),
        "person_seed_ipc_big": " | ".join(sorted(merged_ipc_big)),
        "person_seed_ipc_small": " | ".join(sorted(merged_ipc_small)),
        "person_seed_cpc_expanded": " | ".join(sorted(merged_cpc_expanded)),
        "person_seed_country": " | ".join(sorted(merged_country)),
        "person_seed_years": " | ".join([str(y) for y in sorted(merged_years)]),
        "person_seed_coinventor_sigs": " | ".join(sorted(merged_coinventor_sigs)),
        "_person_profiles_obj": dedup_profiles,
        "_person_assignees_obj": sorted(merged_assignees),
        "_person_ipc_big_obj": sorted(merged_ipc_big),
        "_person_ipc_small_obj": sorted(merged_ipc_small),
        "_person_cpc_expanded_obj": sorted(merged_cpc_expanded),
        "_person_country_obj": sorted(merged_country),
        "_person_years_obj": sorted(merged_years),
        "_person_coinventor_sigs_obj": sorted(merged_coinventor_sigs),
        "_person_seed_patent_ids_obj": sorted(merged_seed_patent_ids),
        "_born_year_obj": born_year,
        "_died_year_obj": died_year,
    })

person_features_df = pd.DataFrame(person_feature_rows)

# =========================================================
# 16. Person-Level Candidate Pool (v3.2)
# =========================================================
person_candidate_rows = []

for _, pf in person_features_df.iterrows():
    person_key = pf["person_key"]
    firstname = pf["firstname"]
    lastname = pf["lastname"]
    inductee = pf["Inductee_example"]

    person_profiles = pf["_person_profiles_obj"]
    person_seed_patent_ids = pf["_person_seed_patent_ids_obj"]
    person_seed_assignees = pf["_person_assignees_obj"]
    born_year = pf["_born_year_obj"]
    died_year = pf["_died_year_obj"]

    lastname_clean = normalize_spaces_and_dot(lastname)
    if lastname_clean == "":
        continue

    # base_sub relaxed:
    # 1) Surname hit
    # 2) Or seed patent hit
    # 3) Or assignee has intersection (bring in roughly first, further filter later by calling coarse_candidate_for_person)
    assignee_hits = group_df["Standardized Assignees"].apply(
        lambda x: assignee_pool_overlap(x, person_seed_assignees)[0]
    )

    base_sub = group_df[
        (
            group_df["Inventors"].astype(str).str.lower().str.contains(re.escape(lastname_clean), na=False)
        ) | (
            group_df["Publication Number Clean"].isin(person_seed_patent_ids)
        ) | (
            assignee_hits
        )
    ].copy()

    for _, pat in base_sub.iterrows():
        pub_no_clean = pat.get("Publication Number Clean", "")
        cand_inventors = pat.get("Inventors", "")
        cand_std_assignee = pat.get("Standardized Assignees", "")
        cand_year = pat.get("Candidate Year", None)

        keep_flag, keep_rule = coarse_candidate_for_person(
            pub_no_clean=pub_no_clean,
            candidate_inventor_field=cand_inventors,
            candidate_std_assignee=cand_std_assignee,
            candidate_year=cand_year,
            firstname=firstname,
            lastname=lastname,
            seed_profiles=person_profiles,
            inductee=inductee,
            person_seed_patent_ids=person_seed_patent_ids,
            person_seed_assignees=person_seed_assignees,
            born_year=born_year,
            died_year=died_year
        )

        if not keep_flag:
            continue

        nm = best_name_match(cand_inventors, person_profiles, firstname, lastname, inductee)

        person_name_score = nm["name_score"]
        person_name_rule = nm["name_rule"]
        matched_seed_name = nm["matched_seed_name"]
        matched_seed_source = nm["matched_seed_source"]
        matched_candidate_name = nm["matched_candidate_name"]

        if keep_rule == "forced_by_seed_patent" and person_name_score == 0:
            person_name_score = 60
            person_name_rule = "forced_by_seed_patent"
            matched_seed_name = "(forced_seed_patent)"
            matched_seed_source = "forced_seed_patent"
            matched_candidate_name = cand_inventors

        person_candidate_rows.append({
            "person_key": person_key,
            "Publication (Announcement) Number": pat.get("Publication (Announcement) Number", ""),
            "Publication Number Clean": pub_no_clean,
            "Publication (Announcement) Date": pat.get("Publication (Announcement) Date", ""),
            "Application Number": pat.get("Application Number", ""),
            "Application Date": pat.get("Application Date", ""),
            "Candidate Year": cand_year,

            "Title (English)": pat.get("Title (English)", ""),
            "Abstract (English)": pat.get("Abstract (English)", ""),
            "Inventors": cand_inventors,
            "First Inventor (Designer)": pat.get("First Inventor (Designer)", ""),
            "Inventor Countries/Regions": pat.get("Inventor Countries/Regions", ""),
            "Applicants": pat.get("Applicants", ""),
            "Standardized Assignees": cand_std_assignee,
            "IPC Main Classification - Major Class": pat.get("IPC Main Classification - Major Class", ""),
            "IPC Main Classification - Subclass": pat.get("IPC Main Classification - Subclass", ""),
            "CPC": pat.get("CPC", ""),

            "person_pool_keep_rule": keep_rule,
            "person_name_score": person_name_score,
            "person_name_rule": person_name_rule,
            "person_matched_seed_name": matched_seed_name,
            "person_matched_seed_source": matched_seed_source,
            "person_matched_candidate_name": matched_candidate_name,
        })

person_candidate_df = pd.DataFrame(person_candidate_rows)

if not person_candidate_df.empty:
    person_candidate_df = person_candidate_df.sort_values(
        by=["person_key", "Publication Number Clean", "person_name_score"],
        ascending=[True, True, False]
    )
    person_candidate_df = person_candidate_df.drop_duplicates(
        subset=["person_key", "Publication Number Clean"],
        keep="first"
    ).reset_index(drop=True)

# =========================================================
# 17. Backfill Candidate Pool to Each No, and Perform Seed-Level Fine Scoring
# =========================================================
rows = []

for _, inv in group_inventors.iterrows():
    no = inv["No"]
    person_key = inv["person_key"]
    inductee = inv["Inductee"]
    firstname = inv["firstname"]
    lastname = inv["lastname"]
    tech = inv["Tech"]

    born_year = extract_year_from_date(inv["Born_year"])
    died_year = extract_year_from_date(inv["Died_year"])

    seed_patent_id = inv.get("seed_patent_id_clean", "")
    seed_profiles = inv.get("_seed_profiles_obj", [])
    seed_assignees = inv.get("_seed_assignees_obj", [])
    seed_ipc_big = inv.get("_seed_ipc_big_obj", [])
    seed_ipc_small = inv.get("_seed_ipc_small_obj", [])
    seed_cpc_expanded = inv.get("_seed_cpc_expanded_obj", [])
    seed_country = inv.get("_seed_country_obj", [])
    seed_years = inv.get("_seed_years_obj", [])
    seed_coinventor_sigs = inv.get("_seed_coinventor_sigs_obj", [])

    sub_candidates = person_candidate_df[person_candidate_df["person_key"] == person_key].copy()

    for _, pat in sub_candidates.iterrows():
        cand_inventors = pat.get("发明人", "")
        cand_assignee_std = pat.get("标准化申请人", "")
        cand_country = pat.get("发明人国家/地区", "")
        cand_year = pat.get("candidate_year", None)
        pub_no_clean = pat.get("公开号_clean", "")

        nm = best_name_match(cand_inventors, seed_profiles, firstname, lastname, inductee)
        name_score = nm["name_score"]

        if name_score <= 0:
            name_score = max(int(pat.get("person_name_score", 0) * 0.6), 0)
            name_rule = f"fallback_from_person_pool:{pat.get('person_name_rule', '')}"
            matched_seed_name = pat.get("person_matched_seed_name", "")
            matched_seed_source = pat.get("person_matched_seed_source", "")
            matched_candidate_name = pat.get("person_matched_candidate_name", "")
        else:
            name_rule = nm["name_rule"]
            matched_seed_name = nm["matched_seed_name"]
            matched_seed_source = nm["matched_seed_source"]
            matched_candidate_name = nm["matched_candidate_name"]

        assignee_score, assignee_rule = assignee_overlap_score(cand_assignee_std, seed_assignees)

        class_score, best_cpc_token, class_rule = class_similarity_score(
            pat.get("IPC主分类-大类", ""),
            pat.get("IPC主分类-小类", ""),
            pat.get("CPC", ""),
            seed_ipc_big,
            seed_ipc_small,
            seed_cpc_expanded
        )

        coauthor_score, co_overlap_n, coauthor_rule = coinventor_overlap_score(
            cand_inventors,
            seed_coinventor_sigs,
            matched_candidate_name
        )

        country_score, country_rule = country_overlap_score(cand_country, seed_country)
        t_score, t_rule = time_score(cand_year, seed_years, born_year, died_year)

        seed_hit = 25 if (seed_patent_id and pub_no_clean == seed_patent_id) else 0

        total_score = (
            name_score
            + assignee_score
            + class_score
            + coauthor_score
            + country_score
            + t_score
            + seed_hit
        )

        rows.append({
            "No": no,
            "person_key": person_key,
            "Inductee": inductee,
            "firstname": firstname,
            "lastname": lastname,
            "Tech": tech,
            "group_id": GROUP_ID,

            "公开（公告）号": pat.get("公开（公告）号", ""),
            "公开号_clean": pub_no_clean,
            "公开（公告）日": pat.get("公开（公告）日", ""),
            "申请号": pat.get("申请号", ""),
            "申请日": pat.get("申请日", ""),
            "candidate_year": cand_year,

            "标题 (英文)": pat.get("标题 (英文)", ""),
            "摘要 (英文)": pat.get("摘要 (英文)", ""),
            "发明人": cand_inventors,
            "第一发明(设计)人": pat.get("第一发明(设计)人", ""),
            "发明人国家/地区": cand_country,
            "申请人": pat.get("申请人", ""),
            "标准化申请人": cand_assignee_std,
            "IPC主分类-大类": pat.get("IPC主分类-大类", ""),
            "IPC主分类-小类": pat.get("IPC主分类-小类", ""),
            "CPC": pat.get("CPC", ""),

            "seed_patent_id_clean": seed_patent_id,
            "seed_patent_found": inv.get("seed_patent_found", False),
            "seed_target_names": inv.get("seed_target_names", ""),
            "seed_assignees": inv.get("seed_assignees", ""),
            "seed_country": inv.get("seed_country", ""),
            "seed_years": inv.get("seed_years", ""),

            "matched_seed_name": matched_seed_name,
            "matched_seed_source": matched_seed_source,
            "matched_candidate_name": matched_candidate_name,

            "person_pool_keep_rule": pat.get("person_pool_keep_rule", ""),
            "person_pool_name_score": pat.get("person_name_score", 0),
            "person_pool_name_rule": pat.get("person_name_rule", ""),

            "name_score": name_score,
            "name_rule": name_rule,

            "assignee_score": assignee_score,
            "assignee_rule": assignee_rule,

            "class_score": class_score,
            "best_cpc_token": best_cpc_token,
            "class_rule": class_rule,

            "coauthor_score": coauthor_score,
            "coauthor_overlap_n": co_overlap_n,
            "coauthor_rule": coauthor_rule,

            "country_score": country_score,
            "country_rule": country_rule,

            "time_score": t_score,
            "time_rule": t_rule,

            "seed_hit": seed_hit,
            "total_score": total_score,
            "match_level": match_level(total_score)
        })

matches_df = pd.DataFrame(rows)

# =========================================================
# 18. Output Organization
# =========================================================
if not matches_df.empty:
    matches_df = matches_df.sort_values(
        by=["No", "total_score", "seed_hit", "name_score", "class_score", "coauthor_score"],
        ascending=[True, False, False, False, False, False]
    ).reset_index(drop=True)

if not matches_df.empty:
    summary_df = (
        matches_df.groupby(["No", "Inductee", "person_key"], as_index=False)
        .agg(
            n_candidates=("公开（公告）号", "count"),
            n_high=("match_level", lambda x: (x == "high").sum()),
            n_medium=("match_level", lambda x: (x == "medium").sum()),
            n_low=("match_level", lambda x: (x == "low").sum()),
            max_score=("total_score", "max"),
            seed_hit_n=("seed_hit", lambda x: (x > 0).sum())
        )
    )
else:
    summary_df = pd.DataFrame(columns=[
        "No", "Inductee", "person_key", "n_candidates", "n_high", "n_medium", "n_low", "max_score", "seed_hit_n"
    ])

if not matches_df.empty:
    person_summary_df = (
        matches_df.groupby(["person_key"], as_index=False)
        .agg(
            n_unique_candidates=("公开号_clean", "nunique"),
            n_rows=("公开（公告）号", "count"),
            n_seed_no=("No", "nunique"),
            example_name=("Inductee", "first")
        )
    )
else:
    person_summary_df = pd.DataFrame(columns=["person_key", "n_unique_candidates", "n_rows", "n_seed_no", "example_name"])

high_df = matches_df[matches_df["match_level"] == "high"].copy() if not matches_df.empty else pd.DataFrame()
medium_df = matches_df[matches_df["match_level"] == "medium"].copy() if not matches_df.empty else pd.DataFrame()

# =========================================================
# 19. Save Results
# =========================================================
group_inventors.drop(
    columns=[c for c in group_inventors.columns if c.startswith("_seed_")],
    errors="ignore"
).to_csv(OUT_DIR / f"group{GROUP_ID}_seed_inventors_v3_2.csv", index=False, encoding="utf-8-sig")

seed_features_df.drop(
    columns=[c for c in seed_features_df.columns if c.startswith("_seed_")],
    errors="ignore"
).to_csv(OUT_DIR / f"group{GROUP_ID}_seed_features_v3_2.csv", index=False, encoding="utf-8-sig")

person_features_df.drop(
    columns=[c for c in person_features_df.columns if c.startswith("_person_")],
    errors="ignore"
).to_csv(OUT_DIR / f"group{GROUP_ID}_person_features_v3_2.csv", index=False, encoding="utf-8-sig")

person_candidate_df.to_csv(OUT_DIR / f"group{GROUP_ID}_person_candidate_pool_v3_2.csv", index=False, encoding="utf-8-sig")
matches_df.to_csv(OUT_DIR / f"group{GROUP_ID}_candidate_matches_v3_2.csv", index=False, encoding="utf-8-sig")
summary_df.to_csv(OUT_DIR / f"group{GROUP_ID}_match_summary_v3_2.csv", index=False, encoding="utf-8-sig")
person_summary_df.to_csv(OUT_DIR / f"group{GROUP_ID}_person_summary_v3_2.csv", index=False, encoding="utf-8-sig")
high_df.to_csv(OUT_DIR / f"group{GROUP_ID}_high_confidence_v3_2.csv", index=False, encoding="utf-8-sig")
medium_df.to_csv(OUT_DIR / f"group{GROUP_ID}_medium_confidence_v3_2.csv", index=False, encoding="utf-8-sig")

print("Done.")
print(f"Outputs saved to: {OUT_DIR}")