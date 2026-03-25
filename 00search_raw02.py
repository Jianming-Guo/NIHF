import re
import pandas as pd

# 读取 incoPat 导出的结果文件
df = pd.read_csv("./data/NIHF_raw/download.csv")

# 假设完整公开号列名为 PN
# 例如：US1569247A, US1569247B2
def extract_core(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r'US(\d+)', s)
    if m:
        return m.group(1)
    return None

df["patent_core"] = df["PN"].apply(extract_core)

# 统计重复主体号
dup_counts = df["patent_core"].value_counts()
dup_cores = dup_counts[dup_counts > 1]

print("重复命中的主体号数量：", len(dup_cores))
print(dup_cores)

# 导出重复记录明细
dup_detail = df[df["patent_core"].isin(dup_cores.index)]
dup_detail.to_csv("./data/NIHF_raw/duplicate_patent_records.csv", index=False, encoding="utf-8-sig")
print("重复记录明细已保存为 duplicate_patent_records.csv")