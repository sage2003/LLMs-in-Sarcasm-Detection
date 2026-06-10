# build_interpretability_sentences.py
import pandas as pd
import numpy as np

DATASETS = {
    "SARC-V2_HYP": "Datasets/SARC-V2/HYP-sarc-notsarc.csv",
    "Llama70B_synthetic": "Datasets/Synthetic/Llama-3.3-70b-versatile/sarcasm_dataset_llama3.3-70b.csv",
    "KimiK2_synthetic": "Datasets/Synthetic/kimi-k2-instruct-0905/sarcasm_dataset_kimi-k2-instruct-0905.csv",
    "SARC-V1": "Datasets/SARC-V1/SARC-V1.csv",
    "SARC-V2_GEN": "Datasets/SARC-V2/GEN-sarc-notsarc.csv",
    "SARC-V2_RQ": "Datasets/SARC-V2/RQ-sarc-notsarc.csv",
    "News_Headlines": "Datasets/News_Headlines/Sarcasm_Headlines_Dataset.csv",
    "Natural-Generated": "Datasets/Intermediate/anchored_fewshot_sarc_llama70b.csv"
}

SEED = 42

def get_text_column(df):
    for col in ["sentence", "text", "headline"]:
        if col in df.columns:
            return col
    raise ValueError

def normalize_label(row):
    for col in ["class", "label"]:
        if col in row and pd.notna(row[col]):
            v = str(row[col]).lower()
            if v in ["sarc", "sarcastic"]:
                return "sarcastic"
            if v in ["notsarc", "not_sarcastic"]:
                return "not_sarcastic"
    return None

rows = []
rng = np.random.RandomState(SEED)

for dname, path in DATASETS.items():
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    text_col = get_text_column(df)
    df["norm_label"] = df.apply(normalize_label, axis=1)
    df = df.dropna(subset=["norm_label"])

    for label in ["sarcastic", "not_sarcastic"]:
        sent = (
            df[df["norm_label"] == label]
            .sample(n=1, random_state=rng.randint(0, 10_000))
            .iloc[0][text_col]
        )
        rows.append({"dataset": dname, "label": label, "sentence": sent})

pd.DataFrame(rows).to_csv("interpretability_sentences.csv", index=False)
print("Saved interpretability_sentences.csv")
