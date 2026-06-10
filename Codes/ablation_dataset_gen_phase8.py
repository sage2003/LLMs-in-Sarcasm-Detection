# run ONCE, not part of Phase 8
import pandas as pd

# df = pd.read_csv("Datasets/SARC-V1/SARC-V1.csv")
# df_fixed = df.sample(n=100, random_state=42)
# df_fixed.to_csv("Datasets/SARC-V1/SARC-V1-ablation_fixed.csv", index=False)


# df = pd.read_csv("Datasets/News_Headlines/Sarcasm_Headlines_Dataset.csv")
# df_fixed = df.sample(n=100, random_state=42)
# df_fixed.to_csv("Datasets/News_Headlines/Sarcasm_Headlines_Dataset_ablation.csv", index=False)

# df = pd.read_csv("Datasets/SARC-V2/HYP-sarc-notsarc.csv")
# df_fixed = df.sample(n=100, random_state=42)
# df_fixed.to_csv("Datasets/SARC-V2/HYP-sarc-notsarc-ablation.csv", index=False)

# df = pd.read_csv("Datasets/Intermediate/anchored_fewshot_sarc_llama70b.csv",)
# df_fixed = df.sample(n=100, random_state=42)
# df_fixed.to_csv("Datasets/Intermediate/anchored_fewshot_sarc_llama70b-ablation.csv", index=False)


import pandas as pd
import os

# ==============================
# CONFIG
# ==============================

SRC_CSV = "Datasets/SARC-V1/SARC-V1.csv"
OUT_DIR = "Datasets/SARC-V1-ablation"
N_SAMPLES = 100          # size of each subset
N_RUNS = 10              # number of subsets to generate
BASE_SEED = 42           # reproducibility anchor

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(SRC_CSV)

# ==============================
# GENERATE SUBSETS
# ==============================

for i in range(N_RUNS):
    seed = BASE_SEED + i

    df_subset = df.sample(
        n=N_SAMPLES,
        random_state=seed
    )

    out_path = f"{OUT_DIR}/sarc_v1_ablation_{i:02d}.csv"
    df_subset.to_csv(out_path, index=False)

    print(f"=== Saved subset {i:02d} | seed={seed} → {out_path} ===")

print("\n=== All ablation subsets generated ===")

