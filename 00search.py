import pandas as pd
import numpy as np
import re

CURRENT_YEAR = 2026

# ========= 1. Load data =========
file_path = r"./data/NIHF_patent_matched.csv"   # Change to your path
df = pd.read_csv(file_path)

# ========= 2. Utility functions =========
def is_missing(x):
    return pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "nan"

def clean_name_part(x):
    """
    Clean name field:
    - Trim leading and trailing spaces
    - Merge multiple spaces
    - Remove punctuation at both ends
    """
    if is_missing(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" .,-;:/")
    return x

def extract_year(x):
    """
    Extract 4-digit year from birth/death date.
    Supported formats:
    - 19500628
    - 1950
    - '1950-06-28'
    - '19500628.0'
    """
    if is_missing(x):
        return None
    
    s = str(x).strip()
    
    # Remove decimal format
    if s.endswith(".0"):
        s = s[:-2]
    
    # Extract first 4-digit year
    m = re.search(r"(17|18|19|20)\d{2}", s)
    if m:
        return int(m.group())
    
    return None

def quote_term(term):
    return f'"{term}"'

def unique_keep_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item not in seen and item != "":
            seen.add(item)
            out.append(item)
    return out

# ========= 3. Name variant generation =========
def generate_name_variants(firstname, lastname, inductee=None):
    firstname = clean_name_part(firstname)
    lastname = clean_name_part(lastname)
    inductee = clean_name_part(inductee) if inductee is not None else ""

    if firstname == "" or lastname == "":
        return []

    # Split firstname into multiple tokens to handle middle names/initials
    first_tokens = firstname.replace(".", " ").split()
    first_tokens = [t for t in first_tokens if t]

    first_main = first_tokens[0] if first_tokens else firstname
    first_initial = first_main[0].upper() if first_main else ""

    variants = []

    # 1) Exact full name
    variants.append(quote_term(f"{firstname} {lastname}"))
    variants.append(quote_term(f"{lastname} {firstname}"))

    # If inductee differs from firstname + lastname, include it
    if inductee and inductee.lower() not in {
        f"{firstname} {lastname}".lower(),
        f"{lastname} {firstname}".lower()
    }:
        variants.append(quote_term(inductee))

    # 2) For multi-part firstname, add main name + lastname only
    if len(first_tokens) >= 2:
        variants.append(quote_term(f"{first_main} {lastname}"))
        variants.append(quote_term(f"{lastname} {first_main}"))

    # 3) Initial-based variants
    if first_initial:
        variants.append(quote_term(f"{first_initial} {lastname}"))
        variants.append(quote_term(f"{lastname} {first_initial}"))

    # 4) Positional search
    variants.append(f"({first_main} (3n) {lastname})")
    variants.append(f"({lastname} (3n) {first_main})")

    if first_initial:
        variants.append(f"({first_initial} (2w) {lastname})")
        variants.append(f"({lastname} (2w) {first_initial})")

    return unique_keep_order(variants)

# ========= 4. Time window generation =========
def generate_time_window(born_raw, died_raw, current_year=CURRENT_YEAR):
    by = extract_year(born_raw)
    dy = extract_year(died_raw)

    # If birth year is missing, do not apply time window to avoid false exclusion
    if by is None:
        return ""

    start_year = by + 15

    if dy is None:
        end_year = min(by + 90, current_year)
    else:
        end_year = min(by + 90, dy, current_year)

    # Prevent invalid year range
    if start_year > end_year:
        return ""

    return f'(AD=[{start_year}0101 to {end_year}1231] OR PD=[{start_year}0101 to {end_year}1231])'

# ========= 5. Generate full query =========
def generate_search_query(row):
    firstname = row.get("firstname", "")
    lastname = row.get("lastname", "")
    inductee = row.get("Inductee", "")
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
    else:
        return base

# ========= 6. Batch generation =========
df["incoPat_query"] = df.apply(generate_search_query, axis=1)

# ========= 7. Export results =========
output_path = r"./data/inventor_incopat_queries.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("Done. Output saved to:", output_path)