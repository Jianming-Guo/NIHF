import pandas as pd
import re
import unicodedata

CURRENT_YEAR = 2026
GROUP_SIZE = 30

# =========================
# 1. Read data (avoid garbled characters as much as possible)
# =========================
def robust_read_csv(file_path):
    """
    Try multiple common encodings to read, avoid special characters being incorrectly converted to '?'
    """
    encodings = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

file_path = r"./data/NIHF_patent_matched.csv"
df = robust_read_csv(file_path)

# =========================
# 2. If fixed columns do not exist, automatically add empty columns
# =========================
for col in ["firstname_fixed", "lastname_fixed", "Inductee_fixed"]:
    if col not in df.columns:
        df[col] = ""

# =========================
# 3. Basic utility functions
# =========================
def is_missing(x):
    return pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan"

def clean_text(x):
    if is_missing(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,-;:/")

def normalize_quotes_dashes(s):
    if not s:
        return ""
    replacements = {
        "’": "'",
        "‘": "'",
        "´": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "‐": "-",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s

def normalize_name_text(x):
    s = clean_text(x)
    s = normalize_quotes_dashes(s)
    return s

def strip_accents(text):
    """
    Remove accent/diacritic marks:
    Brånemark -> Branemark
    José -> Jose
    Müller -> Muller
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text

def make_ascii_safe_name(text):
    """
    Generate a safe name format suitable for incoPat:
    - First normalize
    - Remove accents
    - Keep letters, numbers, spaces, ., -, '
    - Clean up extra spaces
    """
    s = normalize_name_text(text)
    s = strip_accents(s)
    s = re.sub(r"[^A-Za-z0-9\s\.\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_incopat_wildcard_risk(text):
    """
    In incoPat, ? and * have truncation meanings
    """
    if not text:
        return False
    return ("?" in text) or ("*" in text)

def has_bad_replacement_risk(text):
    """
    Rough identification of common garbled/alternative risks
    """
    if not text:
        return False
    return "?" in text

def extract_year(x):
    if is_missing(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    m = re.search(r"(17|18|19|20)\d{2}", s)
    if m:
        return int(m.group())
    return None

def clean_patent_id(x):
    """
    Clean patent_id:
    - Remove spaces
    - Remove .0
    - Keep only numbers and letters
    """
    if is_missing(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9]", "", s)
    return s.upper()

def quote_term(term):
    return f'"{term}"'

def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def safe_quoteable(term):
    """
    Determine if a phrase is suitable to enter "..."
    """
    if not term:
        return False
    if has_incopat_wildcard_risk(term):
        return False
    return True

def add_if_safe_phrase(variants, text):
    text = clean_text(text)
    if safe_quoteable(text):
        variants.append(quote_term(text))

def add_if_safe_proximity(variants, left, right, op):
    left = clean_text(left)
    right = clean_text(right)
    if left and right and (not has_incopat_wildcard_risk(left)) and (not has_incopat_wildcard_risk(right)):
        variants.append(f"({left} {op} {right})")

# =========================
# 4. Prefer to use fixed columns
# =========================
def get_best_name_value(row, raw_col, fixed_col):
    """
    Prefer to use *_fixed columns; if empty, fall back to raw columns
    """
    fixed_val = row.get(fixed_col, "")
    if not is_missing(fixed_val):
        return fixed_val
    return row.get(raw_col, "")

# =========================
# 5. Firstname expansion (nicknames/common formal names)
# =========================
FIRSTNAME_EXPANSION = {
    "don": ["donald"],
    "bob": ["robert"],
    "rob": ["robert"],
    "jim": ["james"],
    "bill": ["william"],
    "will": ["william"],
    "tom": ["thomas"],
    "jack": ["john"],
    "dick": ["richard"],
    "mike": ["michael"],
    "dave": ["david"],
    "ed": ["edward"],
    "ben": ["benjamin"],
    "dan": ["daniel"],
    "sam": ["samuel"],
    "joe": ["joseph"],
    "frank": ["francis", "franklin"],
    "hank": ["henry"],
    "ted": ["theodore", "edward"],
    "chuck": ["charles"],
    "larry": ["lawrence"],
    "harry": ["henry"],
    "walt": ["walter"],
    "alex": ["alexander"],
    "andy": ["andrew"],
    "tony": ["anthony"],
    "pat": ["patrick"],
    "rick": ["richard"],
    "ron": ["ronald"],
    "steve": ["steven", "stephen"],
    "jeff": ["jeffrey"],
    "greg": ["gregory"],
    "ken": ["kenneth"],
    "matt": ["matthew"],
    "chris": ["christopher"],
}

def get_firstname_candidates(firstname):
    """
    For example Don -> ['Don', 'Donald']
    Also add ASCII-safe forms
    """
    firstname = normalize_name_text(firstname)
    if firstname == "":
        return []

    tokens = firstname.replace(".", " ").split()
    tokens = [t for t in tokens if t]
    if not tokens:
        return []

    first_main = tokens[0]
    candidates = [first_main]

    if first_main.lower() in FIRSTNAME_EXPANSION:
        candidates.extend(FIRSTNAME_EXPANSION[first_main.lower()])

    full_firstname = " ".join(tokens)
    if len(tokens) >= 2:
        candidates.append(full_firstname)

    extra = []
    for c in candidates:
        safe_c = make_ascii_safe_name(c)
        if safe_c and safe_c != c:
            extra.append(safe_c)

    candidates.extend(extra)
    return unique_keep_order(candidates)

# =========================
# 6. Name variant generation (prefer fixed columns)
# =========================
def generate_name_variants(firstname, lastname, inductee=None):
    firstname_raw = normalize_name_text(firstname)
    lastname_raw = normalize_name_text(lastname)
    inductee_raw = normalize_name_text(inductee) if inductee else ""

    if firstname_raw == "" or lastname_raw == "":
        return []

    firstname_safe = make_ascii_safe_name(firstname_raw)
    lastname_safe = make_ascii_safe_name(lastname_raw)
    inductee_safe = make_ascii_safe_name(inductee_raw) if inductee_raw else ""

    variants = []

    first_tokens_raw = firstname_raw.replace(".", " ").split()
    first_tokens_raw = [t for t in first_tokens_raw if t]
    if not first_tokens_raw:
        return []

    first_main_raw = first_tokens_raw[0]

    firstname_candidates = get_firstname_candidates(firstname_raw)
    lastname_candidates = unique_keep_order([lastname_raw, lastname_safe])

    # 1. 原始短语（安全时）
    for ln in lastname_candidates:
        add_if_safe_phrase(variants, f"{firstname_raw} {ln}")
        add_if_safe_phrase(variants, f"{ln} {firstname_raw}")

    if inductee_raw:
        add_if_safe_phrase(variants, inductee_raw)
    if inductee_safe and inductee_safe != inductee_raw:
        add_if_safe_phrase(variants, inductee_safe)

    # 2. firstname 多成分时补主名形式
    if len(first_tokens_raw) >= 2:
        for ln in lastname_candidates:
            add_if_safe_phrase(variants, f"{first_main_raw} {ln}")
            add_if_safe_phrase(variants, f"{ln} {first_main_raw}")

    # 3. ASCII-safe 主形式
    if firstname_safe and lastname_safe:
        add_if_safe_phrase(variants, f"{firstname_safe} {lastname_safe}")
        add_if_safe_phrase(variants, f"{lastname_safe} {firstname_safe}")

        first_tokens_safe = firstname_safe.replace(".", " ").split()
        if first_tokens_safe:
            first_main_safe = first_tokens_safe[0]
            first_initial_safe = first_main_safe[0].upper()

            add_if_safe_phrase(variants, f"{first_initial_safe} {lastname_safe}")
            add_if_safe_phrase(variants, f"{lastname_safe} {first_initial_safe}")

            add_if_safe_proximity(variants, first_main_safe, lastname_safe, "(3n)")
            add_if_safe_proximity(variants, lastname_safe, first_main_safe, "(3n)")
            add_if_safe_proximity(variants, first_initial_safe, lastname_safe, "(2w)")
            add_if_safe_proximity(variants, lastname_safe, first_initial_safe, "(2w)")

    # 4. firstname
    for fn in firstname_candidates:
        fn = normalize_name_text(fn)
        fn_safe = make_ascii_safe_name(fn)

        fn_forms = unique_keep_order([fn, fn_safe])

        for fn_form in fn_forms:
            if not fn_form:
                continue

            fn_tokens = fn_form.replace(".", " ").split()
            if not fn_tokens:
                continue

            fn_main = fn_tokens[0]
            fn_initial = fn_main[0].upper()

            for ln in lastname_candidates:
                ln_safe = make_ascii_safe_name(ln)
                ln_forms = unique_keep_order([ln, ln_safe])

                for ln_form in ln_forms:
                    if not ln_form:
                        continue

                    add_if_safe_phrase(variants, f"{fn_form} {ln_form}")
                    add_if_safe_phrase(variants, f"{ln_form} {fn_form}")

                    add_if_safe_phrase(variants, f"{fn_initial} {ln_form}")
                    add_if_safe_phrase(variants, f"{ln_form} {fn_initial}")

                    add_if_safe_proximity(variants, fn_form, ln_form, "(3n)")
                    add_if_safe_proximity(variants, ln_form, fn_form, "(3n)")
                    add_if_safe_proximity(variants, fn_initial, ln_form, "(2w)")
                    add_if_safe_proximity(variants, ln_form, fn_initial, "(2w)")

    return unique_keep_order(variants)

# =========================
# 7. Time window
# =========================
def generate_time_window(born, died):
    by = extract_year(born)
    dy = extract_year(died)

    if by is None:
        return ""

    start_year = by + 15

    if dy is None:
        end_year = min(by + 90, CURRENT_YEAR)
    else:
        end_year = min(by + 90, dy, CURRENT_YEAR)

    if start_year > end_year:
        return ""

    return f'(AD=[{start_year}0101 to {end_year}1231] OR PD=[{start_year}0101 to {end_year}1231])'

# =========================
# 8. query generation for single person (used for pre-check and also can be used as individual query if needed)
# =========================
def generate_single_query(row):
    firstname = get_best_name_value(row, "firstname", "firstname_fixed")
    lastname = get_best_name_value(row, "lastname", "lastname_fixed")
    inductee = get_best_name_value(row, "Inductee", "Inductee_fixed")
    born = row.get("Born_year", "")
    died = row.get("Died_year", "")

    name_variants = generate_name_variants(firstname, lastname, inductee)
    if not name_variants:
        return ""

    name_block = " OR ".join(name_variants)
    time_block = generate_time_window(born, died)

    base = f'pt=(4) AND PNC=US AND IN=({name_block})'
    if time_block:
        return f'{base} AND {time_block}'
    return base

# =========================
# 9. Summary block
# =========================
def generate_person_block(row):
    firstname = get_best_name_value(row, "firstname", "firstname_fixed")
    lastname = get_best_name_value(row, "lastname", "lastname_fixed")
    inductee = get_best_name_value(row, "Inductee", "Inductee_fixed")
    born = row.get("Born_year", "")
    died = row.get("Died_year", "")

    name_variants = generate_name_variants(firstname, lastname, inductee)
    if not name_variants:
        return ""

    name_block = " OR ".join(name_variants)
    time_block = generate_time_window(born, died)

    if time_block:
        return f'(IN=({name_block}) AND {time_block})'
    else:
        return f'(IN=({name_block}))'

# =========================
# 10. Generate normalized fields (raw columns, fixed columns, final adopted columns)
# =========================
df["firstname_final"] = df.apply(lambda r: get_best_name_value(r, "firstname", "firstname_fixed"), axis=1)
df["lastname_final"] = df.apply(lambda r: get_best_name_value(r, "lastname", "lastname_fixed"), axis=1)
df["Inductee_final"] = df.apply(lambda r: get_best_name_value(r, "Inductee", "Inductee_fixed"), axis=1)

df["firstname_norm"] = df["firstname_final"].apply(normalize_name_text)
df["lastname_norm"] = df["lastname_final"].apply(normalize_name_text)
df["inductee_norm"] = df["Inductee_final"].apply(normalize_name_text)

df["firstname_ascii"] = df["firstname_norm"].apply(make_ascii_safe_name)
df["lastname_ascii"] = df["lastname_norm"].apply(make_ascii_safe_name)
df["inductee_ascii"] = df["inductee_norm"].apply(make_ascii_safe_name)

df["used_fixed_firstname"] = df["firstname_fixed"].apply(lambda x: not is_missing(x))
df["used_fixed_lastname"] = df["lastname_fixed"].apply(lambda x: not is_missing(x))
df["used_fixed_inductee"] = df["Inductee_fixed"].apply(lambda x: not is_missing(x))

df["name_has_wildcard_risk"] = df.apply(
    lambda r: has_incopat_wildcard_risk(r.get("firstname_norm", "")) or
              has_incopat_wildcard_risk(r.get("lastname_norm", "")) or
              has_incopat_wildcard_risk(r.get("inductee_norm", "")),
    axis=1
)

df["name_has_bad_replacement_risk"] = df.apply(
    lambda r: has_bad_replacement_risk(r.get("firstname_norm", "")) or
              has_bad_replacement_risk(r.get("lastname_norm", "")) or
              has_bad_replacement_risk(r.get("inductee_norm", "")),
    axis=1
)

# =========================
# 11. Generate query
# =========================
df["name_variants_list"] = df.apply(
    lambda r: generate_name_variants(
        r.get("firstname_final", ""),
        r.get("lastname_final", ""),
        r.get("Inductee_final", "")
    ),
    axis=1
)

df["name_variant_count"] = df["name_variants_list"].apply(len)
df["name_variants_preview"] = df["name_variants_list"].apply(lambda x: " | ".join(x[:12]))

df["single_query"] = df.apply(generate_single_query, axis=1)
df["person_block"] = df.apply(generate_person_block, axis=1)

# =========================
# 12. Group by No every 30 entries
# =========================
df["group_id"] = ((df["No"] - 1) // GROUP_SIZE) + 1

# =========================
# 13. Generate grouped query
# =========================
group_rows = []
for gid, sub in df.groupby("group_id"):
    blocks = [x for x in sub["person_block"].tolist() if x]
    group_query = "pt=(4) AND PNC=US AND (\n" + "\nOR\n".join(blocks) + "\n)"
    group_rows.append({
        "group_id": gid,
        "n_inventors": len(sub),
        "group_query": group_query
    })
group_query_df = pd.DataFrame(group_rows)

# =========================
# 14. Pre-check fields
# =========================
df["patent_id_clean"] = df["patent_id"].apply(clean_patent_id) if "patent_id" in df.columns else ""
df["born_year_extracted"] = df["Born_year"].apply(extract_year) if "Born_year" in df.columns else None
df["died_year_extracted"] = df["Died_year"].apply(extract_year) if "Died_year" in df.columns else None

df["has_patent_id"] = df["patent_id_clean"].apply(lambda x: x != "")
df["has_birth_year"] = df["born_year_extracted"].notna()
df["has_time_window"] = df.apply(
    lambda r: generate_time_window(r.get("Born_year", ""), r.get("Died_year", "")) != "",
    axis=1
)
df["has_name_query"] = df["person_block"].apply(lambda x: x != "")
df["query_ready"] = df["has_name_query"]

df["manual_name_repair_needed"] = df.apply(
    lambda r: (
        r.get("name_has_bad_replacement_risk", False)
        or (r.get("firstname_norm", "") != "" and r.get("firstname_ascii", "") == "")
        or (r.get("lastname_norm", "") != "" and r.get("lastname_ascii", "") == "")
    ),
    axis=1
)

def make_risk_flag(row):
    flags = []

    if not row["has_name_query"]:
        flags.append("missing_name_query")

    if not row["has_patent_id"]:
        flags.append("missing_patent_id")

    if not row["has_birth_year"]:
        flags.append("missing_birth_year")

    if row["has_birth_year"] and not row["has_time_window"]:
        flags.append("invalid_time_window")

    lastname = clean_text(row.get("lastname_final", ""))
    firstname = clean_text(row.get("firstname_final", ""))

    if len(lastname) <= 2:
        flags.append("short_lastname")

    if len(firstname) <= 4 and len(lastname) <= 3:
        flags.append("common_short_name_risk")

    if row.get("name_variant_count", 0) < 8:
        flags.append("few_name_variants")

    if row.get("name_has_wildcard_risk", False):
        flags.append("name_contains_incopat_wildcard_char")

    if row.get("name_has_bad_replacement_risk", False):
        flags.append("name_encoding_replacement_risk")

    if row.get("manual_name_repair_needed", False):
        flags.append("manual_name_repair_needed")

    return ";".join(flags)

df["risk_flag"] = df.apply(make_risk_flag, axis=1)

# =========================
# 15. Export
# =========================
output_df = df.drop(columns=["name_variants_list"], errors="ignore").copy()

output_df.to_csv("./data/inventor_query_mapping_with_check.csv", index=False, encoding="utf-8-sig")
group_query_df.to_csv("./data/grouped_queries_30_each.csv", index=False, encoding="utf-8-sig")

risk_df = output_df[output_df["risk_flag"] != ""].copy()
risk_df.to_csv("./data/inventor_risk_checklist.csv", index=False, encoding="utf-8-sig")

print("Done.")
print("Saved files:")
print("- ./data/inventor_query_mapping_with_check.csv")
print("- ./data/grouped_queries_30_each.csv")
print("- ./data/inventor_risk_checklist.csv")