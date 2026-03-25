import pandas as pd

# ==========================
# 1 Read CSV
# ==========================
input_file = "./data/NIHF_patent_matched.csv"
clean_file = "./data/NIHF_raw/clean_patents.csv"
query_file = "./data/NIHF_raw/incopat_query.txt"

df = pd.read_csv(input_file)

# ==========================
# 2 Process patent_id
# ==========================
def format_patent_id(x):
    if pd.isna(x):
        return None
    try:
        x = str(int(float(x)))  # 去掉 .0
        return x
    except:
        return None

df["patent_id_clean"] = df["patent_id"].apply(format_patent_id)

# ==========================
# 3 Remove NaN and deduplicate
# ==========================
patents = df["patent_id_clean"].dropna().unique()

# ==========================
# 4 Generate search query
# ==========================
queries = []

for p in patents:
    queries.append(f"US{p}A*")
    queries.append(f"US{p}B*")

query = "PN=(" + " OR ".join(queries) + ")"

# ==========================
# 5 Save cleaned patent list
# ==========================
df_clean = pd.DataFrame({"patent_id": patents})
df_clean.to_csv(clean_file, index=False)

# ==========================
# 6 Save search query
# ==========================
with open(query_file, "w", encoding="utf-8") as f:
    f.write(query)

print("Processing completed")
print("Number of patents generated:", len(patents))
print("Search query example:")
print(query[:500] + ("..." if len(query) > 500 else ""))