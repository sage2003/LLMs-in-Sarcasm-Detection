import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# ================= CONFIG =================

DATASETS = {
    "Llama70B_synthetic": "../Datasets/Synthetic/Llama-3.3-70b-versatile/sarcasm_dataset_llama3.3-70b.csv",
    "KimiK2_synthetic": "../Datasets/Synthetic/kimi-k2-instruct-0905/sarcasm_dataset_kimi-k2-instruct-0905.csv",
    "SARC-V1": "../Datasets/SARC-V1/SARC-V1.csv",
    "SARC-V2_HYP": "../Datasets/SARC-V2/HYP-sarc-notsarc.csv",
    "SARC-V2_GEN": "../Datasets/SARC-V2/GEN-sarc-notsarc.csv",
    "SARC-V2_RQ": "../Datasets/SARC-V2/RQ-sarc-notsarc.csv",
    "News_Headlines": "../Datasets/News_Headlines/Sarcasm_Headlines_Dataset.csv",
    "Natural-Generated": "../Datasets/Intermediate/anchored_fewshot_sarc_llama70b.csv"
}

OUTPUT_CSV = "../Outputs/dataset_summary.csv"
OUTPUT_LATEX = "../Outputs/dataset_summary.tex"
OUTPUT_PNG = "Final_Plots/dataset_summary.png"

# ==========================================


# -------- Detect text column --------
def get_text_column(df):
    for col in ["sentence", "text", "headline"]:
        if col in df.columns:
            return col
    raise ValueError("No valid text column found.")


# -------- Tokenizer --------
def tokenize(text):
    return re.findall(r"\b\w+\b", str(text).lower())


# -------- Compute stats --------
def compute_stats(df, text_col):
    texts = df[text_col].dropna().astype(str)

    N = len(texts)

    all_tokens = []
    lengths = []

    for t in texts:
        tokens = tokenize(t)
        lengths.append(len(tokens))
        all_tokens.extend(tokens)

    avg_len = np.mean(lengths) if lengths else 0

    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0

    return N, avg_len, ttr


# -------- Distribution label --------
def infer_distribution(name):
    name = name.lower()

    if "kimi" in name or "llama" in name or "generated" in name:
        return "Synthetic"
    elif "sarc" in name or "news" in name:
        return "Organic"
    else:
        return "Mixed"

def normalize_label(val):
    if pd.isna(val):
        return None
    v = str(val).lower()
    if v in ["sarc", "sarcastic", "1"]:
        return "sarcastic"
    elif v in ["notsarc", "not_sarcastic", "0"]:
        return "not_sarcastic"
    return None

def save_table_as_png(df, output_path):
    fig, ax = plt.subplots(figsize=(14, len(df) * 0.6 + 1))  # wider figure
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Adjust column widths (key fix)
    col_widths = [0.18, 0.12, 0.12, 0.14, 0.12, 0.14, 0.12]

    for col, width in enumerate(col_widths):
        for row in range(len(df) + 1):
            table[(row, col)].set_width(width)

    # Bold header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    # Left-align dataset column (cleaner)
    for row in range(1, len(df) + 1):
        table[(row, 0)]._loc = 'left'
        table[(row, 0)].PAD = 0.02

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" === Saved {output_path} ===")

# -------- Main --------
def main():
    rows = []

    for name, path in DATASETS.items():
        print(f" === Processing {name} ===")

        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        text_col = get_text_column(df)

        # -------- Label column detection --------
        label_col = "class" if "class" in df.columns else ("label" if "label" in df.columns else None)

        sarcastic_count = 0
        nonsarcastic_count = 0

        if label_col is not None:
            labels = df[label_col].apply(normalize_label)

            sarcastic_count = (labels == "sarcastic").sum()
            nonsarcastic_count = (labels == "not_sarcastic").sum()

        N, avg_len, ttr = compute_stats(df, text_col)

        rows.append({
            "Dataset": name,
            "Size (N)": f"{N:,}",
            "# Sarcastic": sarcastic_count,
            "# Not Sarcastic": nonsarcastic_count,
            "Avg. Length": round(avg_len, 1),
            "Type-Token Ratio": round(ttr, 2),
            "Distribution": infer_distribution(name)
        })

    table_df = pd.DataFrame(rows)

    # -------- Save CSV --------
    table_df.to_csv(OUTPUT_CSV, index=False)
    print(f" === Saved {OUTPUT_CSV} ===")

    # -------- Save LaTeX --------
    latex = table_df.to_latex(index=False, escape=False)
    with open(OUTPUT_LATEX, "w") as f:
        f.write(latex)

    print(f" === Saved {OUTPUT_LATEX} ===")

    # -------- Print --------
    print("\n === FINAL TABLE === \n")
    print(table_df.to_string(index=False))

    save_table_as_png(table_df, OUTPUT_PNG)


if __name__ == "__main__":
    main()