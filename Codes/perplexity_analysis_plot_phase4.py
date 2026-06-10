import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= USER CONFIG =================
MODELS = [
    "llama-3.1-8b-instruct",
    "qwen2.5-14b-instruct",
    "phi-3-medium-4k-instruct",
    "gemma-2-9b-it"
]

CSV_DIR = "../CSV"
OUT_DIR = "Final_Plots"

# visual tweaks
FIGSIZE = (12, 6)
LINEWIDTH = 2.5
FONTSIZE_LABEL = 16
FONTSIZE_TITLE = 22
FONTSIZE_LEGEND = 12
# ==============================================

os.makedirs(OUT_DIR, exist_ok=True)


def plot_histogram(model_name):
    df = pd.read_csv(f"{CSV_DIR}/perplexity_hist_avg_{model_name}.csv")

    plt.figure(figsize=FIGSIZE)
    colors = sns.color_palette("tab10", n_colors=df["dataset"].nunique())
    datasets = sorted(df["dataset"].unique())

    for color, dname in zip(colors, datasets):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            subset = df[(df["dataset"] == dname) & (df["label"] == label)]

            plt.plot(
                subset["x"],
                subset["y"],
                linestyle=style,
                color=color,
                linewidth=LINEWIDTH,
                label=f"{dname} ({label})"
            )

    plt.yscale("log")
    plt.xlabel("Perplexity", fontsize=FONTSIZE_LABEL)
    plt.ylabel("Density (averaged)", fontsize=FONTSIZE_LABEL)
    plt.title(f"Perplexity Histogram\nModel: {model_name}", fontsize=FONTSIZE_TITLE)

    plt.legend(fontsize=FONTSIZE_LEGEND, ncol=2)
    plt.tight_layout()

    plt.savefig(f"{OUT_DIR}/hist_{model_name}.png", dpi=500)
    plt.savefig(f"{OUT_DIR}/hist_{model_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_kde(model_name):
    df = pd.read_csv(f"{CSV_DIR}/logppl_kde_avg_{model_name}.csv")

    plt.figure(figsize=FIGSIZE)
    colors = sns.color_palette("tab10", n_colors=df["dataset"].nunique())
    datasets = sorted(df["dataset"].unique())

    for color, dname in zip(colors, datasets):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            subset = df[(df["dataset"] == dname) & (df["label"] == label)]

            plt.plot(
                subset["log_perplexity"],
                subset["density"],
                linestyle=style,
                color=color,
                linewidth=LINEWIDTH,
                label=f"{dname} ({label})"
            )

    plt.xlabel("log(Perplexity)", fontsize=FONTSIZE_LABEL)
    plt.ylabel("Density (averaged)", fontsize=FONTSIZE_LABEL)
    plt.title(f"Averaged Log-Perplexity KDE\nModel: {model_name}", fontsize=FONTSIZE_TITLE)

    plt.legend(fontsize=FONTSIZE_LEGEND, ncol=2)
    plt.tight_layout()

    plt.savefig(f"{OUT_DIR}/kde_{model_name}.png", dpi=500)
    plt.savefig(f"{OUT_DIR}/kde_{model_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_truncated_hist(model_name):
    df = pd.read_csv(f"{CSV_DIR}/trunc_perplexity_hist_avg_{model_name}.csv")

    plt.figure(figsize=FIGSIZE)
    colors = sns.color_palette("tab10", n_colors=df["dataset"].nunique())
    datasets = sorted(df["dataset"].unique())

    for color, dname in zip(colors, datasets):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            subset = df[(df["dataset"] == dname) & (df["label"] == label)]

            plt.plot(
                subset["x"],
                subset["y"],
                linestyle=style,
                color=color,
                linewidth=LINEWIDTH,
                label=f"{dname} ({label})"
            )

    plt.yscale("log")
    plt.xlabel("Perplexity (≤ 5000)", fontsize=FONTSIZE_LABEL)
    plt.ylabel("Density (averaged)", fontsize=FONTSIZE_LABEL)
    plt.title(f"Truncated Perplexity Histogram\nModel: {model_name}", fontsize=FONTSIZE_TITLE)

    plt.legend(fontsize=FONTSIZE_LEGEND, ncol=2)
    plt.tight_layout()

    plt.savefig(f"{OUT_DIR}/trunc_hist_{model_name}.png", dpi=500)
    plt.savefig(f"{OUT_DIR}/trunc_hist_{model_name}.pdf", bbox_inches="tight")
    plt.close()


# ================= RUN =================
if __name__ == "__main__":
    for model in MODELS:
        print(f" === Plotting for {model} ===")
        #plot_histogram(model)
        plot_kde(model)
        #plot_truncated_hist(model)

    print("\n === All plots generated ===")