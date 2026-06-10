"""
Compute per-sentence perplexity for multiple LLMs over multiple datasets
and generate a single histogram per model (all datasets combined).
"""

import torch
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import seaborn as sns
from scipy.stats import gaussian_kde


# ================= USER CONFIG =================
DATASETS = {
    "SARC-V2_HYP": "../Datasets/SARC-V2/HYP-sarc-notsarc.csv",
    "Llama70B_synthetic": "../Datasets/Synthetic/Llama-3.3-70b-versatile/sarcasm_dataset_llama3.3-70b.csv",
    "KimiK2_synthetic": "../Datasets/Synthetic/kimi-k2-instruct-0905/sarcasm_dataset_kimi-k2-instruct-0905.csv",
    "SARC-V1": "../Datasets/SARC-V1/SARC-V1.csv",
    "SARC-V2_GEN": "../Datasets/SARC-V2/GEN-sarc-notsarc.csv",
    "SARC-V2_RQ": "../Datasets/SARC-V2/RQ-sarc-notsarc.csv",
    "News_Headlines": "../Datasets/News_Headlines/Sarcasm_Headlines_Dataset.csv",
    "Natural-Generated": "../Datasets/Intermediate/anchored_fewshot_sarc_llama70b.csv"
}




MODELS = {
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "phi-3-medium-4k-instruct": "microsoft/Phi-3-medium-4k-instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
}

BATCH_SIZE = 20
MAX_LENGTH = 256
USE_4BIT = True
SAMPLE_SIZE = 1100
RANDOM_SEED = 42
N_RUNS = 100
# ==============================================


def get_text_column(df):
    """Automatically find the text column."""
    for col in ["sentence", "text", "headline"]:
        if col in df.columns:
            return col
    raise ValueError("No valid text column found.")


def compute_sentence_ppl(model, tokenizer, sentences):
    """Compute per-sentence perplexity (correctly)."""
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]
        labels = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    mask = attention_mask[:, 1:]
    token_log_probs = token_log_probs * mask

    sent_nll = -token_log_probs.sum(dim=1) / mask.sum(dim=1)
    sent_ppl = torch.exp(sent_nll)

    return sent_ppl.cpu().tolist()

def normalize_label(row):
    # natural datasets
    if "class" in row:
        if row["class"] == "sarc":
            return "sarcastic"
        elif row["class"] == "notsarc":
            return "not_sarcastic"

    # synthetic datasets
    if "class" in row:
        if row["class"] in ["sarcastic", "sarc"]:
            return "sarcastic"
        elif row["class"] in ["not_sarcastic", "notsarc"]:
            return "not_sarcastic"

    return None

# ================= MAIN LOOP =================
for model_name, model_id in MODELS.items():
    print(f"\n === Loading model: {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    ) if USE_4BIT else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",      #balanced
        quantization_config=bnb_config
    ).eval()

    all_runs = []

    # -------- Dataset loop --------
    for run in range(N_RUNS):
        print(f"\n === Run {run + 1}/{N_RUNS} ===")
        run_dfs = []
        for dname, dpath in DATASETS.items():
            print(f" === Processing dataset: {dname} ===")

            df =pd.read_csv(
                dpath,
                engine="python",
                on_bad_lines="skip"
                )

            # ---- Random sampling to control dataset size ----
            if len(df) > SAMPLE_SIZE:
                df = df.sample(
                    n=SAMPLE_SIZE,
                    random_state=RANDOM_SEED + run
                ).reset_index(drop=True)
                print(f"  ↳ Randomly sampled {SAMPLE_SIZE} rows")
            else:
                print(f"  ⚠ Dataset has only {len(df)} rows; using all")

            text_col = get_text_column(df)
            sentences = df[text_col].astype(str).tolist()

            ppls = []
            for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
                batch = sentences[i:i + BATCH_SIZE]
                ppls.extend(compute_sentence_ppl(model, tokenizer, batch))

            tmp = df.copy()
            tmp["perplexity"] = ppls
            tmp["dataset"] = dname
            tmp["model"] = model_name
            run_dfs.append(tmp)
        run_df = pd.concat(run_dfs, ignore_index=True)
        all_runs.append(run_df)
        assert len(run_dfs) == len(DATASETS), "Some datasets missing in run"
    assert len(all_runs) == N_RUNS, "Mismatch in number of runs"



    # -------- Save CSV --------
    final_df = pd.concat(all_runs, ignore_index=True)
    csv_name = f"../Outputs/Phase-4/1100_points/perplexity_all_datasets_{model_name}_1100_runs.csv"
    final_df.to_csv(csv_name, index=False)
    print(f" === Saved CSV: {csv_name} ===")

    # ---- Clean perplexity values globally ----
    before = len(final_df)

    final_df = final_df[np.isfinite(final_df["perplexity"])]

    after = len(final_df)
    print(f"Removed {before - after} rows with inf/NaN perplexity")

    final_df["norm_label"] = final_df.apply(normalize_label, axis=1)
    final_df = final_df.dropna(subset=["norm_label"])

    # -------- CSV: cleaned final values --------
    clean_csv_name = f"../CSV/perplexity_clean_{model_name}.csv"
    final_df.to_csv(clean_csv_name, index=False)
    print(f" === Saved cleaned CSV: {clean_csv_name} ===")
 
    bins = np.linspace(0, final_df["perplexity"].max(), 80)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist_accumulator = {
        (dname, label): np.zeros(len(bins)-1)
        for dname in DATASETS
        for label in ["sarcastic", "not_sarcastic"]
    }


    for run_df in all_runs:
        run_df = run_df[np.isfinite(run_df["perplexity"])].copy()
        run_df["norm_label"] = run_df.apply(normalize_label, axis=1)
        run_df = run_df.dropna(subset=["norm_label"])

        for dname in DATASETS:
            for label in ["sarcastic", "not_sarcastic"]:
                vals = run_df[
                    (run_df["dataset"] == dname) &
                    (run_df["norm_label"] == label)
                ]["perplexity"]

                hist, _ = np.histogram(
                    vals,
                    bins=bins,
                    density=True
                )

                hist_accumulator[(dname, label)] += hist

    # -------- CSV: averaged histogram --------
    hist_rows = []
    for dname in DATASETS:
        for label in ["sarcastic", "not_sarcastic"]:
            avg_hist = hist_accumulator[(dname, label)] / N_RUNS
            for x, y in zip(bin_centers, avg_hist):
                hist_rows.append({
                    "model": model_name,
                    "dataset": dname,
                    "label": label,
                    "perplexity_bin": x,
                    "density": y
                })

    hist_df = pd.DataFrame(hist_rows)
    hist_csv_name = f"../CSV/perplexity_hist_avg_{model_name}.csv"
    hist_df.to_csv(hist_csv_name, index=False)
    print(f" === Saved histogram CSV: {hist_csv_name} ===")            

    # -------- Plot Histogram --------
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("tab10", n_colors=len(DATASETS))

    for color, dname in zip(colors, DATASETS):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            avg_hist = hist_accumulator[(dname, label)] / N_RUNS
            plt.plot(
                bin_centers,
                avg_hist,
                linestyle=style,
                color=color,
                linewidth=2,
                label=f"{dname} ({label})"
            )

    plt.yscale("log")
    plt.xlabel("Perplexity")
    plt.ylabel("Density (averaged)")
    plt.title(f"Averaged Perplexity Histogram ({N_RUNS} random subsamples)\nModel: {model_name}")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"../Outputs/Phase-4/1100_points/perplexity_hist_avg_{model_name}({N_RUNS} runs)_1100pts_all_datasets.png", dpi=500)
    plt.savefig(f"../Outputs/Phase-4/1100_points/perplexity_hist_avg_{model_name}({N_RUNS} runs)_1100pts_all_datasets.pdf", dpi=500, bbox_inches="tight")
    plt.close()

    print(" === Saved linear-PPL histogram ===")

    # log-perplexity for plotting
    final_df["log_ppl"] = np.log(final_df["perplexity"])

    # ===== Averaged KDE over N_RUNS =====

    plt.figure(figsize=(12, 6))

    datasets = sorted(final_df["dataset"].unique())
    colors = sns.color_palette("tab10", n_colors=len(datasets))

    # fixed grid for KDE evaluation (VERY important)
    x_grid = np.linspace(
        final_df["log_ppl"].min(),
        final_df["log_ppl"].max(),
        400
    )

    # accumulator: (dataset, label) -> summed KDE
    kde_accumulator = {
        (dname, label): np.zeros_like(x_grid)
        for dname in datasets
        for label in ["sarcastic", "not_sarcastic"]
    }

    # ---- accumulate KDEs over runs ----
    for run_df in all_runs:   # all_runs = list of per-run DataFrames
        run_df = run_df[np.isfinite(run_df["perplexity"])].copy()
        run_df["norm_label"] = run_df.apply(normalize_label, axis=1)
        run_df = run_df.dropna(subset=["norm_label"])
        run_df["log_ppl"] = np.log(run_df["perplexity"])

        for dname in datasets:
            for label in ["sarcastic", "not_sarcastic"]:
                vals = run_df[
                    (run_df["dataset"] == dname) &
                    (run_df["norm_label"] == label)
                ]["log_ppl"].values

                if len(vals) < 2:
                    continue  # KDE undefined for <2 points

                kde = gaussian_kde(vals)
                kde_accumulator[(dname, label)] += kde(x_grid)

    # -------- CSV: averaged KDE --------
    kde_rows = []
    for dname in datasets:
        for label in ["sarcastic", "not_sarcastic"]:
            avg_kde = kde_accumulator[(dname, label)] / N_RUNS
            for x, y in zip(x_grid, avg_kde):
                kde_rows.append({
                    "model": model_name,
                    "dataset": dname,
                    "label": label,
                    "log_perplexity": x,
                    "density": y
                })

    kde_df = pd.DataFrame(kde_rows)
    kde_csv_name = f"../CSV/logppl_kde_avg_{model_name}.csv"
    kde_df.to_csv(kde_csv_name, index=False)
    print(f" === Saved KDE CSV: {kde_csv_name} ===")

    # ---- plot averaged KDEs ----
    for color, dname in zip(colors, datasets):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            avg_kde = kde_accumulator[(dname, label)] / N_RUNS

            plt.plot(
                x_grid,
                avg_kde,
                color=color,
                linestyle=style,
                linewidth=2,
                label=f"{dname} ({label})"
            )

    plt.xlabel("log(Perplexity)")
    plt.ylabel("Density (averaged)")
    plt.title(
        f"Averaged Log-Perplexity KDE ({N_RUNS} random subsamples)\n"
        f"Model: {model_name}"
    )

    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"../Outputs/Phase-4/1100_points/logppl_kde_avg_{model_name}({N_RUNS} runs)_1100pts_all_datasets.png", dpi=500)
    plt.savefig(f"../Outputs/Phase-4/1100_points/logppl_kde_avg_{model_name}({N_RUNS} runs)_1100pts_all_datasets.pdf", bbox_inches="tight")
    plt.close()

    print(" === Saved averaged log-PPL KDE ===")

    # truncated histogram plot

    PPL_MAX = 5000
    bins = np.linspace(0, PPL_MAX, 80)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    hist_accumulator = {
        (dname, label): np.zeros(len(bins)-1)
        for dname in DATASETS
        for label in ["sarcastic", "not_sarcastic"]
    }


    for run_df in all_runs:
        run_df = run_df[np.isfinite(run_df["perplexity"])].copy()
        run_df["norm_label"] = run_df.apply(normalize_label, axis=1)
        run_df = run_df.dropna(subset=["norm_label"])

        for dname in DATASETS:
            for label in ["sarcastic", "not_sarcastic"]:
                vals = run_df[
                    (run_df["dataset"] == dname) &
                    (run_df["norm_label"] == label)
                ]["perplexity"]

                vals = vals[vals <= PPL_MAX]

                hist, _ = np.histogram(
                    vals,
                    bins=bins,
                    density=True
                )

                hist_accumulator[(dname, label)] += hist
    
    # -------- CSV: truncated histogram --------
    trunc_rows = []
    for dname in DATASETS:
        for label in ["sarcastic", "not_sarcastic"]:
            avg_hist = hist_accumulator[(dname, label)] / N_RUNS
            for x, y in zip(bin_centers, avg_hist):
                trunc_rows.append({
                    "model": model_name,
                    "dataset": dname,
                    "label": label,
                    "perplexity_bin": x,
                    "density": y
                })

    trunc_df = pd.DataFrame(trunc_rows)
    trunc_csv_name = f"../CSV/trunc_perplexity_hist_avg_{model_name}.csv"
    trunc_df.to_csv(trunc_csv_name, index=False)
    print(f" === Saved truncated histogram CSV: {trunc_csv_name} ===")

    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("tab10", n_colors=len(DATASETS))

    # visualization-only cap (does NOT affect analysis)
    plot_df = final_df[final_df["perplexity"] <= PPL_MAX]


    for color, dname in zip(colors, DATASETS):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            avg_hist = hist_accumulator[(dname, label)] / N_RUNS
            plt.plot(
                bin_centers,
                avg_hist,
                linestyle=style,
                color=color,
                linewidth=2,
                label=f"{dname} ({label})"
            )

    plt.yscale("log")
    plt.xlabel("Perplexity")
    plt.ylabel("Density (averaged)")
    plt.title(f"Averaged Perplexity Histogram ({N_RUNS} random subsamples)\nModel: {model_name}")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"../Outputs/Phase-4/1100_points/trunc_perplexity_hist_avg_{model_name}({N_RUNS} runs)_1100pts_all_datasets.png", dpi=500)
    plt.savefig(f"../Outputs/Phase-4/1100_points/trunc_perplexity_hist_avg_{model_name}({N_RUNS} runs)_1100pts_all_datasets.pdf", dpi=500, bbox_inches="tight")
    plt.close()

    print(" === Saved linear-PPL histogram ===")

    # -------- Cleanup GPU memory --------
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

print("\n === All models processed successfully ===")
