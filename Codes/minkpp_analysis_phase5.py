"""
Min-K%++ membership analysis over multiple LLMs and datasets
with optional Monte-Carlo runs and perplexity-style plots.
"""

import math
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.stats import gaussian_kde

# ================= USER CONFIG =================
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


MODELS = {
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "phi-3-medium-4k-instruct": "microsoft/Phi-3-medium-4k-instruct",  

}


BATCH_SIZE = 16
MAX_LENGTH = 256
K_PERCENT = 0.10

USE_4BIT = True
DTYPE = torch.float16

# Monte-Carlo control
SAMPLE_SIZE = 1100      # set to e.g. 1164 or None
N_RUNS = 100              # >1 only makes sense if SAMPLE_SIZE != None
RANDOM_SEED = 42

TOP_K_PRINT = 10
# ==============================================


# ---------- Helpers ----------
def get_text_column(df):
    for col in ["sentence", "text", "headline"]:
        if col in df.columns:
            return col
    raise ValueError("No valid text column found.")


def normalize_label(row):
    if "class" in row and pd.notna(row["class"]):
        v = str(row["class"]).lower()
        if v in ["sarc", "sarcastic", "sarcastics"]:
            return "sarcastic"
        if v in ["notsarc", "not_sarcastic", "not-sarcastic"]:
            return "not_sarcastic"

    if "label" in row and pd.notna(row["label"]):
        v = str(row["label"]).lower()
        if v in ["sarc", "sarcastic", "sarcastics"]:
            return "sarcastic"
        if v in ["notsarc", "not_sarcastic", "not-sarcastic"]:
            return "not_sarcastic"

    return None


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = None
    if USE_4BIT:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_cfg,
        low_cpu_mem_usage=True,
    ).eval()

    return tokenizer, model


def compute_minkpp_batch(model, tokenizer, sentences):
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        logits = model(input_ids).logits

    logits = logits[:, :-1]
    targets = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    token_logp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    mu = (probs * log_probs).sum(dim=-1)
    sigma2 = (probs * log_probs**2).sum(dim=-1) - mu**2
    sigma = torch.sqrt(torch.clamp(sigma2, min=1e-20))

    z = (token_logp - mu) / sigma
    mask = attention_mask[:, 1:].bool()

    scores = []
    for i in range(z.size(0)):
        vals = z[i][mask[i]].cpu().numpy()
        if len(vals) == 0:
            scores.append(0.0)
            continue
        k = max(1, int(math.ceil(len(vals) * K_PERCENT)))
        scores.append(float(np.mean(np.partition(vals, k - 1)[:k])))

    return scores


# ================= MAIN =================
for model_name, model_id in MODELS.items():
    print(f"\n === Model: {model_name} ===")
    tokenizer, model = load_model(model_id)

    all_runs = []

    for run in range(N_RUNS):
        print(f"\n === Run {run+1}/{N_RUNS} ===")
        run_dfs = []

        for dname, dpath in DATASETS.items():
            print(f" === {dname} ===")
            df = pd.read_csv(dpath, engine="python", on_bad_lines="skip")

            if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
                df = df.sample(
                    n=SAMPLE_SIZE,
                    random_state=RANDOM_SEED + run
                ).reset_index(drop=True)

            text_col = get_text_column(df)
            df = df.copy()
            df["__text__"] = df[text_col].astype(str)

            sentences = df["__text__"].tolist()

            scores = []
            for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
                batch = sentences[i:i+BATCH_SIZE]
                scores.extend(compute_minkpp_batch(model, tokenizer, batch))

            tmp = df.copy()
            tmp["minkpp_score"] = scores
            tmp["dataset"] = dname
            tmp["run"] = run
            run_dfs.append(tmp)

        run_df = pd.concat(run_dfs, ignore_index=True)
        all_runs.append(run_df)

    # ---------- Aggregate ----------
    final_df = pd.concat(all_runs, ignore_index=True)
    final_df["norm_label"] = final_df.apply(normalize_label, axis=1)
    final_df = final_df.dropna(subset=["norm_label"]).copy()

    agg_df = (
        final_df
        .groupby(["dataset", "norm_label", "__text__"], as_index=False)
        .agg(mean_minkpp=("minkpp_score", "mean"))
        .rename(columns={"__text__": "text"})
    )

    agg_df["rank"] = agg_df["mean_minkpp"].rank(
        method="min", ascending=False
    ).astype(int)

    csv_name = f"../Outputs/minkpp_all_datasets_{model_name}_avg_{SAMPLE_SIZE}.csv"
    agg_df.to_csv(csv_name, index=False)
    print(f" === Saved {csv_name} ===")

    # ---------- TOP-K ----------
    print(f"\nTop {TOP_K_PRINT} suspected memorized sentences:")
    for _, r in agg_df.sort_values("mean_minkpp", ascending=False).head(TOP_K_PRINT).iterrows():
        print(f"rank={r['rank']}\tscore={r['mean_minkpp']:.4f}\t{r['text'][:200]}")

    # ================= PLOTTING =================
    datasets = sorted(agg_df["dataset"].unique())
    colors = sns.color_palette("tab10", n_colors=len(datasets))

    # ---------- Histogram ----------
    plt.figure(figsize=(12, 6))
    # -------- CSV: histogram (final plotted values) --------
    hist_rows = []      ######

    SCORE_MIN, SCORE_MAX = -8, 1
    bins = np.linspace(SCORE_MIN, SCORE_MAX, 80)
    centers = 0.5 * (bins[:-1] + bins[1:])

    has_artist = False
    for color, dname in zip(colors, datasets):
        df_d = agg_df[agg_df["dataset"] == dname]
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            vals = df_d[df_d["norm_label"] == label]["mean_minkpp"].values
            if len(vals) < 2:
                continue
            hist, _ = np.histogram(vals, bins=bins, density=True)
            for x, y in zip(centers, hist):
                hist_rows.append({
                    "model": model_name,
                    "dataset": dname,
                    "label": label,
                    "minkpp_bin": x,
                    "density": y
                })

            plt.plot(centers, hist, color=color, linestyle=style, linewidth=2,
                     label=f"{dname} ({label})")
            has_artist = True

    hist_df = pd.DataFrame(hist_rows)
    hist_csv_name = f"../CSV/minkpp_hist_avg_{model_name}.csv"
    hist_df.to_csv(hist_csv_name, index=False)
    print(f" === Saved histogram CSV: {hist_csv_name} ===")

    if has_artist:
        plt.legend(fontsize=8, ncol=2)


    plt.xlabel("Min-K%++ score")
    plt.ylabel("Density")
    plt.title(f"Min-K%++ Histogram\nModel: {model_name}")
    plt.tight_layout()
    plt.savefig(f"../Outputs/n_minkpp_hist_{model_name}_avg_{SAMPLE_SIZE}.png", dpi=500)
    plt.close()

    # ---------- KDE ----------
    plt.figure(figsize=(12, 6))

    # -------- CSV: KDE (final plotted values) --------
    kde_rows = []

    x_grid = np.linspace(-25, 2, 400)

    has_artist = False
    for color, dname in zip(colors, datasets):
        df_d = agg_df[agg_df["dataset"] == dname]
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            vals = df_d[df_d["norm_label"] == label]["mean_minkpp"].values
            vals = vals[np.isfinite(vals)]
            if len(vals) < 2:
                continue
            kde = gaussian_kde(vals)
            y_vals = kde(x_grid)

            for x, y in zip(x_grid, y_vals):
                kde_rows.append({
                    "model": model_name,
                    "dataset": dname,
                    "label": label,
                    "minkpp_score": x,
                    "density": y
                })

            plt.plot(x_grid, kde(x_grid), color=color, linestyle=style,
                     linewidth=2, label=f"{dname} ({label})")
            has_artist = True
  
    kde_df = pd.DataFrame(kde_rows)
    kde_csv_name = f"../CSV/minkpp_kde_avg_{model_name}.csv"
    kde_df.to_csv(kde_csv_name, index=False)
    print(f" === Saved KDE CSV: {kde_csv_name} ===")
    
    if has_artist:
        plt.legend(fontsize=8, ncol=2)

    plt.xlabel("Min-K%++ score")
    plt.ylabel("Density")
    plt.title(f"Min-K%++ KDE\nModel: {model_name}")
    plt.tight_layout()
    plt.savefig(f"../Outputs/n_minkpp_kde_{model_name}_avg_{SAMPLE_SIZE}.png", dpi=500)
    plt.close()

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

print("\n === Min-K%++ analysis completed successfully ===")
