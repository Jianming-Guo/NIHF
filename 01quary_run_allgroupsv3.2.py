import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

# =========================================================
# 0. 参数设置
# =========================================================
CURRENT_YEAR = 2026

BASE = Path("./data")

FILE_MATCHED = BASE / "NIHF_patent_matched.csv"
FILE_MAPPING = BASE / "inventor_query_mapping_with_check.csv"
FILE_SEED_RAW = BASE / "NIHF_raw" / "NIHF_raw_download.xlsx"
DIR_GROUP_FILES = BASE / "NIHF_all"

OUT_BASE = BASE / "04_match_results_v3_2"
OUT_GROUPS = OUT_BASE / "groups"
OUT_MERGED = OUT_BASE / "merged"

OUT_GROUPS.mkdir(parents=True, exist_ok=True)
OUT_MERGED.mkdir(parents=True, exist_ok=True)

# =========================================================
# 1. 基础工具函数
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

def safe_join(seq, sep=" | "):
    """
    安全拼接：
    - 跳过 NaN / None / 空字符串
    - 统一转 str
    """
    out = []
    for x in seq:
        if pd.isna(x):
            continue
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            continue
        out.append(s)
    return sep.join(out)

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

def build_person_key(firstname, lastname, born_year):
    fn = normalize_spaces_and_dot(firstname)
    ln = normalize_spaces_and_dot(lastname)
    by = extract_year_from_date(born_year)
    if by is not None:
        return f"{fn}||{ln}||{by}"
    return f"{fn}||{ln}"

# =========================================================
# 2. NIHF_all 中下载结果采用英文列名
#    NIHF_raw_download.xlsx 仍按原中文列名读取
# =========================================================
# group download 英文列名
COL_NO = "NO"
COL_PUB_NO = "Publication NO."
COL_PUB_DATE = "Publication Date"
COL_APP_NO = "Application NO."
COL_APP_DATE = "Application Date"
COL_APP_DATE_ALT = "Application Date "
COL_PATENT_TYPE = "Patent Type"
COL_PUB_AUTH = "Publication Authority"
COL_TITLE_EN = "Title (English)"
COL_ABS_EN = "Abstract (English)"
COL_INVENTOR = "Inventor"
COL_INVENTOR_ALT = "Inventor "
COL_FIRST_INVENTOR = "First Inventor/Designer"
COL_INVENTOR_N = "Inventor/Designer Number"
COL_INVENTOR_COUNTRY = "Inventor Country/Area"
COL_INVENTOR_ADDR = "Inventor Address (Original)"
COL_APPLICANT = "Applicant"
COL_APPLICANT_ALT = "Applicant "
COL_NORM_APPLICANT = "Normalized Applicant"
COL_NORM_APPLICANT_ALT = "Normalized Applicant "
COL_APPLICANT_COUNTRY = "Applicant Country/Area Code"
COL_APPLICANT_ADDR = "Applicant Address"
COL_APPLICANT_N = "Applicant Number"
COL_IPC_MAIN_CLASS = "IPC Main Class"
COL_IPC_MAIN_SUBCLASS = "IPC Main Subclass"
COL_IPC_MAIN_GROUP = "IPC Main Group"
COL_IPC_MAIN_SUBGROUP = "IPC Main Subgroup"
COL_CPC = "CPC"
COL_CIT = "Citation Number of Times"
COL_FWD_CIT = "Citation-Forward Times"
COL_SELF_CIT = "Self-citing Times"
COL_NONSELF_CIT = "Non-self-citing Times"
COL_SIMPLE_FAM_ID = "Simple Family ID"
COL_SIMPLE_FAM_N = "Simple Family Number"
COL_DOCDB_FAM_ID = "DocDB Family ID"
COL_FIRST_CLAIM = "First Claim"

def get_col(df, preferred, alt=None):
    if preferred in df.columns:
        return preferred
    if alt and alt in df.columns:
        return alt
    return preferred

# seed raw 中文列名
SEED_COL_PUB_NO = "公开（公告）号"
SEED_COL_PUB_DATE = "公开（公告）日"
SEED_COL_APP_NO = "申请号"
SEED_COL_APP_DATE = "申请日"
SEED_COL_INVENTOR = "发明人"
SEED_COL_NORM_APPLICANT = "标准化申请人"
SEED_COL_INVENTOR_COUNTRY = "发明人国家/地区"
SEED_COL_IPC_MAIN_CLASS = "IPC主分类-大类"
SEED_COL_IPC_MAIN_SUBCLASS = "IPC主分类-小类"
SEED_COL_CPC = "CPC"

# =========================================================
# 3. 姓名等价（昵称 / 正式名）双向映射
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
# 4. 姓名 profile 构建
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
# 5. 姓名比较
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
# 6. 申请人交集函数
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
    score, rule = assignee_overlap_score(candidate_std_assignee, seed_assignees)
    return score > 0, rule

# =========================================================
# 7. 共发明人 / 国家地区
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
        return 12, "ipc_subclass_overlap"
    if c_big and c_big in seed_big:
        return 7, "ipc_class_overlap"
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
# 9. 时间评分
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
    if candidate_year is None or pd.isna(candidate_year):
        return False

    candidate_year = int(candidate_year)

    if born_year is None:
        return True

    start_y = born_year + 15
    end_y = min(born_year + 90, CURRENT_YEAR) if died_year is None else min(born_year + 90, died_year, CURRENT_YEAR)

    return start_y <= candidate_year <= end_y

def match_level(total_score):
    if total_score >= 75:
        return "high"
    elif total_score >= 50:
        return "medium"
    else:
        return "low"

# =========================================================
# 10. person-level 粗回填规则（v3.2）
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
    if pub_no_clean and pub_no_clean in set(person_seed_patent_ids):
        return True, "forced_by_seed_patent"

    lastname_clean = normalize_spaces_and_dot(lastname)
    text = clean_text(candidate_inventor_field)

    if lastname_clean and lastname_clean in text:
        if len(lastname_clean) >= 4:
            return True, "lastname_only_pool"

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

    assignee_keep, assignee_rule = assignee_pool_overlap(candidate_std_assignee, person_seed_assignees)
    if assignee_keep and time_window_ok(candidate_year, born_year, died_year):
        return True, f"assignee_pool_fallback:{assignee_rule}"

    return False, "not_in_person_pool"

# =========================================================
# 11. 文件扫描与分组
# =========================================================
def discover_group_files(group_dir: Path):
    """
    扫描 ./data/NIHF_all 下所有 xlsx
    以文件名前缀数字识别 group_id
    例如:
      18a.xlsx -> 18
      23b.xlsx -> 23
      1.xlsx   -> 1
    """
    files = sorted(group_dir.glob("*.xlsx"))
    group_map = defaultdict(list)

    for fp in files:
        stem = fp.stem.strip()
        m = re.match(r"^(\d+)", stem)
        if m:
            gid = int(m.group(1))
            group_map[gid].append(fp)

    return dict(sorted(group_map.items(), key=lambda kv: kv[0]))

# =========================================================
# 12. 读取并合并某个 group 的多个下载文件
# =========================================================
def load_group_downloads(group_files):
    dfs = []
    for fp in group_files:
        temp = pd.read_excel(fp)
        temp["source_file"] = fp.name
        dfs.append(temp)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    return out

# =========================================================
# 13. 单个 group 的 v3.2 运行函数
# =========================================================
def run_single_group_v3_2(group_id, group_files, matched_df, mapping_df, seed_raw_df):
    print(f"\n===== Running group {group_id} =====")
    print("Files:")
    for fp in group_files:
        print(f"  - {fp.name}")

    group_out_dir = OUT_GROUPS / f"group_{group_id:03d}"
    group_out_dir.mkdir(parents=True, exist_ok=True)

    group_df = load_group_downloads(group_files)
    if group_df.empty:
        print(f"[WARN] Group {group_id} has no valid rows.")
        return None

    # group 英文列名兼容
    g_inventor = get_col(group_df, COL_INVENTOR, COL_INVENTOR_ALT)
    g_pub_no = get_col(group_df, COL_PUB_NO)
    g_pub_date = get_col(group_df, COL_PUB_DATE)
    g_app_no = get_col(group_df, COL_APP_NO)
    g_app_date = get_col(group_df, COL_APP_DATE, COL_APP_DATE_ALT)
    g_title = get_col(group_df, COL_TITLE_EN)
    g_abs = get_col(group_df, COL_ABS_EN)
    g_first_inventor = get_col(group_df, COL_FIRST_INVENTOR)
    g_inventor_country = get_col(group_df, COL_INVENTOR_COUNTRY)
    g_applicant = get_col(group_df, COL_APPLICANT, COL_APPLICANT_ALT)
    g_norm_applicant = get_col(group_df, COL_NORM_APPLICANT, COL_NORM_APPLICANT_ALT)
    g_ipc_class = get_col(group_df, COL_IPC_MAIN_CLASS)
    g_ipc_subclass = get_col(group_df, COL_IPC_MAIN_SUBCLASS)
    g_ipc_group = get_col(group_df, COL_IPC_MAIN_GROUP)
    g_ipc_subgroup = get_col(group_df, COL_IPC_MAIN_SUBGROUP)
    g_cpc = get_col(group_df, COL_CPC)

    group_df["pub_no_clean"] = group_df[g_pub_no].apply(patent_numeric_core)
    group_df["application_year"] = group_df[g_app_date].apply(extract_year_from_date)
    group_df["publication_year"] = group_df[g_pub_date].apply(extract_year_from_date)
    group_df["candidate_year"] = group_df["application_year"].fillna(group_df["publication_year"])
    group_df["group_id"] = group_id

    group_inventors = mapping_df[mapping_df["group_id"] == group_id].copy()
    if group_inventors.empty:
        print(f"[WARN] Group {group_id} has no inventors in mapping file.")
        return None

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

    seed_raw_local = seed_raw_df.copy()
    seed_raw_local["pub_no_clean"] = seed_raw_local[SEED_COL_PUB_NO].apply(patent_numeric_core)
    seed_raw_local["application_year"] = seed_raw_local[SEED_COL_APP_DATE].apply(extract_year_from_date)
    seed_raw_local["publication_year"] = seed_raw_local[SEED_COL_PUB_DATE].apply(extract_year_from_date)

    # -------------------------
    # No-level seed features
    # -------------------------
    seed_feature_rows = []

    for _, inv in group_inventors.iterrows():
        no = inv["No"]
        person_key = inv["person_key"]
        inductee = inv["Inductee"]
        firstname = inv["firstname"]
        lastname = inv["lastname"]
        patent_id_clean = patent_numeric_core(inv["patent_id"])

        seed_patents = seed_raw_local[seed_raw_local["pub_no_clean"] == patent_id_clean].copy()

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
            inv_field = "" if pd.isna(r.get(SEED_COL_INVENTOR, "")) else str(r.get(SEED_COL_INVENTOR, ""))
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

            for a in split_multi_value(r.get(SEED_COL_NORM_APPLICANT, "")):
                seed_assignees.add(normalize_org(a))

            for c in split_multi_value(r.get(SEED_COL_INVENTOR_COUNTRY, "")):
                seed_country.add(clean_text(c))

            big = clean_text(r.get(SEED_COL_IPC_MAIN_CLASS, ""))
            small = clean_text(r.get(SEED_COL_IPC_MAIN_SUBCLASS, ""))
            if big:
                seed_ipc_big.add(big)
            if small:
                seed_ipc_small.add(small)

            seed_cpc_expanded.update(expand_cpc_field(r.get(SEED_COL_CPC, "")))

            if pd.notna(r.get("application_year", None)):
                seed_years.append(int(r["application_year"]))
            if pd.notna(r.get("publication_year", None)):
                seed_years.append(int(r["publication_year"]))

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
            "seed_inventor_field_raw": safe_join(unique_keep_order(seed_inventor_field_raw)),
            "seed_target_names": safe_join([p["raw_name"] for p in seed_profiles]),
            "seed_target_name_sources": safe_join([f"{p['source']}:{p['strength']}" for p in seed_profiles]),
            "seed_assignees": safe_join(sorted(seed_assignees)),
            "seed_ipc_big": safe_join(sorted(seed_ipc_big)),
            "seed_ipc_small": safe_join(sorted(seed_ipc_small)),
            "seed_cpc_expanded": safe_join(sorted(seed_cpc_expanded)),
            "seed_country": safe_join(sorted(seed_country)),
            "seed_years": safe_join([str(y) for y in sorted(set(seed_years))]),
            "seed_coinventor_sigs": safe_join(sorted(seed_coinventor_sigs)),
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

    # -------------------------
    # person-level features
    # -------------------------
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
            "seed_no_list": safe_join([str(x) for x in seed_no_list], sep="|"),
            "person_seed_patent_ids": safe_join(sorted(merged_seed_patent_ids), sep="|"),
            "person_seed_names": safe_join(unique_keep_order(merged_seed_names)),
            "person_seed_assignees": safe_join(sorted(merged_assignees)),
            "person_seed_ipc_big": safe_join(sorted(merged_ipc_big)),
            "person_seed_ipc_small": safe_join(sorted(merged_ipc_small)),
            "person_seed_cpc_expanded": safe_join(sorted(merged_cpc_expanded)),
            "person_seed_country": safe_join(sorted(merged_country)),
            "person_seed_years": safe_join([str(y) for y in sorted(merged_years)]),
            "person_seed_coinventor_sigs": safe_join(sorted(merged_coinventor_sigs)),
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

    # -------------------------
    # person-level candidate pool
    # -------------------------
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

        assignee_hits = group_df[g_norm_applicant].apply(
            lambda x: assignee_pool_overlap(x, person_seed_assignees)[0]
        )

        base_sub = group_df[
            (
                group_df[g_inventor].astype(str).str.lower().str.contains(re.escape(lastname_clean), na=False)
            ) | (
                group_df["pub_no_clean"].isin(person_seed_patent_ids)
            ) | (
                assignee_hits
            )
        ].copy()

        for _, pat in base_sub.iterrows():
            pub_no_clean = pat.get("pub_no_clean", "")
            cand_inventors = "" if pd.isna(pat.get(g_inventor, "")) else str(pat.get(g_inventor, ""))
            cand_std_assignee = "" if pd.isna(pat.get(g_norm_applicant, "")) else str(pat.get(g_norm_applicant, ""))
            cand_year = pat.get("candidate_year", None)

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
                "group_id": group_id,
                "source_file": pat.get("source_file", ""),
                "Publication NO.": pat.get(g_pub_no, ""),
                "pub_no_clean": pub_no_clean,
                "Publication Date": pat.get(g_pub_date, ""),
                "Application NO.": pat.get(g_app_no, ""),
                "Application Date": pat.get(g_app_date, ""),
                "candidate_year": cand_year,

                "Title (English)": pat.get(g_title, ""),
                "Abstract (English)": pat.get(g_abs, ""),
                "Inventor": cand_inventors,
                "First Inventor/Designer": pat.get(g_first_inventor, ""),
                "Inventor Country/Area": pat.get(g_inventor_country, ""),
                "Applicant": pat.get(g_applicant, ""),
                "Normalized Applicant": cand_std_assignee,
                "IPC Main Class": pat.get(g_ipc_class, ""),
                "IPC Main Subclass": pat.get(g_ipc_subclass, ""),
                "IPC Main Group": pat.get(g_ipc_group, ""),
                "IPC Main Subgroup": pat.get(g_ipc_subgroup, ""),
                "CPC": pat.get(g_cpc, ""),

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
            by=["person_key", "pub_no_clean", "person_name_score"],
            ascending=[True, True, False]
        )
        person_candidate_df = person_candidate_df.drop_duplicates(
            subset=["person_key", "pub_no_clean"],
            keep="first"
        ).reset_index(drop=True)

    # -------------------------
    # 回填到每个 No，并做 seed-level 精评分
    # -------------------------
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
            cand_inventors = "" if pd.isna(pat.get("Inventor", "")) else str(pat.get("Inventor", ""))
            cand_assignee_std = "" if pd.isna(pat.get("Normalized Applicant", "")) else str(pat.get("Normalized Applicant", ""))
            cand_country = pat.get("Inventor Country/Area", "")
            cand_year = pat.get("candidate_year", None)
            pub_no_clean = pat.get("pub_no_clean", "")

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
                pat.get("IPC Main Class", ""),
                pat.get("IPC Main Subclass", ""),
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
                "group_id": group_id,

                "source_file": pat.get("source_file", ""),
                "Publication NO.": pat.get("Publication NO.", ""),
                "pub_no_clean": pub_no_clean,
                "Publication Date": pat.get("Publication Date", ""),
                "Application NO.": pat.get("Application NO.", ""),
                "Application Date": pat.get("Application Date", ""),
                "candidate_year": cand_year,

                "Title (English)": pat.get("Title (English)", ""),
                "Abstract (English)": pat.get("Abstract (English)", ""),
                "Inventor": cand_inventors,
                "First Inventor/Designer": pat.get("First Inventor/Designer", ""),
                "Inventor Country/Area": cand_country,
                "Applicant": pat.get("Applicant", ""),
                "Normalized Applicant": cand_assignee_std,
                "IPC Main Class": pat.get("IPC Main Class", ""),
                "IPC Main Subclass": pat.get("IPC Main Subclass", ""),
                "IPC Main Group": pat.get("IPC Main Group", ""),
                "IPC Main Subgroup": pat.get("IPC Main Subgroup", ""),
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

    if not matches_df.empty:
        matches_df = matches_df.sort_values(
            by=["No", "total_score", "seed_hit", "name_score", "class_score", "coauthor_score"],
            ascending=[True, False, False, False, False, False]
        ).reset_index(drop=True)

    if not matches_df.empty:
        summary_df = (
            matches_df.groupby(["No", "Inductee", "person_key"], as_index=False)
            .agg(
                n_candidates=("Publication NO.", "count"),
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
                n_unique_candidates=("pub_no_clean", "nunique"),
                n_rows=("Publication NO.", "count"),
                n_seed_no=("No", "nunique"),
                example_name=("Inductee", "first")
            )
        )
    else:
        person_summary_df = pd.DataFrame(columns=["person_key", "n_unique_candidates", "n_rows", "n_seed_no", "example_name"])

    high_df = matches_df[matches_df["match_level"] == "high"].copy() if not matches_df.empty else pd.DataFrame()
    medium_df = matches_df[matches_df["match_level"] == "medium"].copy() if not matches_df.empty else pd.DataFrame()

    # -------------------------
    # 保存本组结果
    # -------------------------
    group_inventors.drop(
        columns=[c for c in group_inventors.columns if c.startswith("_seed_")],
        errors="ignore"
    ).to_csv(group_out_dir / f"group_{group_id:03d}_seed_inventors_v3_2.csv", index=False, encoding="utf-8-sig")

    seed_features_df.drop(
        columns=[c for c in seed_features_df.columns if c.startswith("_seed_")],
        errors="ignore"
    ).to_csv(group_out_dir / f"group_{group_id:03d}_seed_features_v3_2.csv", index=False, encoding="utf-8-sig")

    person_features_df.drop(
        columns=[c for c in person_features_df.columns if c.startswith("_person_")],
        errors="ignore"
    ).to_csv(group_out_dir / f"group_{group_id:03d}_person_features_v3_2.csv", index=False, encoding="utf-8-sig")

    group_df.to_csv(group_out_dir / f"group_{group_id:03d}_download_raw_merged.csv", index=False, encoding="utf-8-sig")
    person_candidate_df.to_csv(group_out_dir / f"group_{group_id:03d}_person_candidate_pool_v3_2.csv", index=False, encoding="utf-8-sig")
    matches_df.to_csv(group_out_dir / f"group_{group_id:03d}_candidate_matches_v3_2.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(group_out_dir / f"group_{group_id:03d}_match_summary_v3_2.csv", index=False, encoding="utf-8-sig")
    person_summary_df.to_csv(group_out_dir / f"group_{group_id:03d}_person_summary_v3_2.csv", index=False, encoding="utf-8-sig")
    high_df.to_csv(group_out_dir / f"group_{group_id:03d}_high_confidence_v3_2.csv", index=False, encoding="utf-8-sig")
    medium_df.to_csv(group_out_dir / f"group_{group_id:03d}_medium_confidence_v3_2.csv", index=False, encoding="utf-8-sig")

    return {
        "group_id": group_id,
        "download_raw_merged": group_df,
        "seed_inventors": group_inventors.drop(columns=[c for c in group_inventors.columns if c.startswith("_seed_")], errors="ignore"),
        "seed_features": seed_features_df.drop(columns=[c for c in seed_features_df.columns if c.startswith("_seed_")], errors="ignore"),
        "person_features": person_features_df.drop(columns=[c for c in person_features_df.columns if c.startswith("_person_")], errors="ignore"),
        "person_candidate_pool": person_candidate_df,
        "candidate_matches": matches_df,
        "match_summary": summary_df,
        "person_summary": person_summary_df,
        "high_confidence": high_df,
        "medium_confidence": medium_df,
    }

# =========================================================
# 14. 主程序：跑全部 groups
# =========================================================
def main():
    print("Loading core files...")
    matched_df = pd.read_csv(FILE_MATCHED)
    mapping_df = pd.read_csv(FILE_MAPPING)
    seed_raw_df = pd.read_excel(FILE_SEED_RAW)

    print("Scanning NIHF_all group files...")
    group_file_map = discover_group_files(DIR_GROUP_FILES)

    if not group_file_map:
        print("[ERROR] No group xlsx files found in ./data/NIHF_all/")
        return

    print("Detected groups:")
    for gid, fps in group_file_map.items():
        print(f"  Group {gid}: {[fp.name for fp in fps]}")

    all_download_raw = []
    all_seed_inventors = []
    all_seed_features = []
    all_person_features = []
    all_person_candidate_pool = []
    all_candidate_matches = []
    all_match_summary = []
    all_person_summary = []
    all_high_confidence = []
    all_medium_confidence = []

    run_log_rows = []

    for group_id, group_files in group_file_map.items():
        try:
            result = run_single_group_v3_2(
                group_id=group_id,
                group_files=group_files,
                matched_df=matched_df,
                mapping_df=mapping_df,
                seed_raw_df=seed_raw_df
            )

            if result is None:
                run_log_rows.append({
                    "group_id": group_id,
                    "status": "skipped_or_empty",
                    "n_files": len(group_files),
                    "files": " | ".join([fp.name for fp in group_files])
                })
                continue

            if result["download_raw_merged"] is not None and not result["download_raw_merged"].empty:
                all_download_raw.append(result["download_raw_merged"])

            if result["seed_inventors"] is not None and not result["seed_inventors"].empty:
                all_seed_inventors.append(result["seed_inventors"])

            if result["seed_features"] is not None and not result["seed_features"].empty:
                all_seed_features.append(result["seed_features"])

            if result["person_features"] is not None and not result["person_features"].empty:
                all_person_features.append(result["person_features"])

            if result["person_candidate_pool"] is not None and not result["person_candidate_pool"].empty:
                all_person_candidate_pool.append(result["person_candidate_pool"])

            if result["candidate_matches"] is not None and not result["candidate_matches"].empty:
                all_candidate_matches.append(result["candidate_matches"])

            if result["match_summary"] is not None and not result["match_summary"].empty:
                all_match_summary.append(result["match_summary"])

            if result["person_summary"] is not None and not result["person_summary"].empty:
                all_person_summary.append(result["person_summary"])

            if result["high_confidence"] is not None and not result["high_confidence"].empty:
                all_high_confidence.append(result["high_confidence"])

            if result["medium_confidence"] is not None and not result["medium_confidence"].empty:
                all_medium_confidence.append(result["medium_confidence"])

            run_log_rows.append({
                "group_id": group_id,
                "status": "ok",
                "n_files": len(group_files),
                "files": " | ".join([fp.name for fp in group_files]),
                "n_download_rows": len(result["download_raw_merged"]) if result["download_raw_merged"] is not None else 0,
                "n_match_rows": len(result["candidate_matches"]) if result["candidate_matches"] is not None else 0
            })

        except Exception as e:
            run_log_rows.append({
                "group_id": group_id,
                "status": f"error: {str(e)}",
                "n_files": len(group_files),
                "files": " | ".join([fp.name for fp in group_files])
            })
            print(f"[ERROR] group {group_id} failed: {e}")

    def safe_concat(dfs):
        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    merged_download_raw = safe_concat(all_download_raw)
    merged_seed_inventors = safe_concat(all_seed_inventors)
    merged_seed_features = safe_concat(all_seed_features)
    merged_person_features = safe_concat(all_person_features)
    merged_person_candidate_pool = safe_concat(all_person_candidate_pool)
    merged_candidate_matches = safe_concat(all_candidate_matches)
    merged_match_summary = safe_concat(all_match_summary)
    merged_person_summary = safe_concat(all_person_summary)
    merged_high_confidence = safe_concat(all_high_confidence)
    merged_medium_confidence = safe_concat(all_medium_confidence)

    run_log_df = pd.DataFrame(run_log_rows)

    merged_download_raw.to_csv(OUT_MERGED / "all_download_raw_merged_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_seed_inventors.to_csv(OUT_MERGED / "all_seed_inventors_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_seed_features.to_csv(OUT_MERGED / "all_seed_features_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_person_features.to_csv(OUT_MERGED / "all_person_features_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_person_candidate_pool.to_csv(OUT_MERGED / "all_person_candidate_pool_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_candidate_matches.to_csv(OUT_MERGED / "all_candidate_matches_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_match_summary.to_csv(OUT_MERGED / "all_match_summary_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_person_summary.to_csv(OUT_MERGED / "all_person_summary_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_high_confidence.to_csv(OUT_MERGED / "all_high_confidence_v3_2.csv", index=False, encoding="utf-8-sig")
    merged_medium_confidence.to_csv(OUT_MERGED / "all_medium_confidence_v3_2.csv", index=False, encoding="utf-8-sig")
    run_log_df.to_csv(OUT_MERGED / "run_log_v3_2.csv", index=False, encoding="utf-8-sig")

    print("\nAll groups finished.")
    print(f"Group outputs: {OUT_GROUPS}")
    print(f"Merged outputs: {OUT_MERGED}")

if __name__ == "__main__":
    main()