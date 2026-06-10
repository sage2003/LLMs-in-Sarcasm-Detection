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
FONTSIZE_LEGEND = 8

# X-axis limits (None = auto)
X_RANGE = (-8,1)   # e.g., (-2, 3)
# ==============================================

os.makedirs(OUT_DIR, exist_ok=True)


def plot_histogram(model_name):
    df = pd.read_csv(f"{CSV_DIR}/minkpp_hist_avg_{model_name}.csv")

    plt.figure(figsize=FIGSIZE)
    colors = sns.color_palette("tab10", n_colors=df["dataset"].nunique())
    datasets = sorted(df["dataset"].unique())

    for color, dname in zip(colors, datasets):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            subset = df[(df["dataset"] == dname) & (df["label"] == label)]

            plt.plot(
                subset["minkpp_bin"],
                subset["density"],
                linestyle=style,
                color=color,
                linewidth=LINEWIDTH,
                label=f"{dname} ({label})"
            )
    if X_RANGE is not None:
        plt.xlim(X_RANGE)
    plt.xlabel("Min-K++ Score", fontsize=FONTSIZE_LABEL)
    plt.ylabel("Density", fontsize=FONTSIZE_LABEL)
    plt.title(f"Min-K++ Histogram\nModel: {model_name}", fontsize=FONTSIZE_TITLE)

    plt.legend(fontsize=FONTSIZE_LEGEND, ncol=2)
    plt.tight_layout()

    plt.savefig(f"{OUT_DIR}/minkpp_hist_{model_name}.png", dpi=500)
    plt.savefig(f"{OUT_DIR}/minkpp_hist_{model_name}.pdf", bbox_inches="tight")
    plt.close()


def plot_kde(model_name):
    df = pd.read_csv(f"{CSV_DIR}/minkpp_kde_avg_{model_name}.csv")

    plt.figure(figsize=FIGSIZE)
    colors = sns.color_palette("tab10", n_colors=df["dataset"].nunique())
    datasets = sorted(df["dataset"].unique())

    for color, dname in zip(colors, datasets):
        for label, style in [("sarcastic", "-"), ("not_sarcastic", "--")]:
            subset = df[(df["dataset"] == dname) & (df["label"] == label)]

            plt.plot(
                subset["minkpp_score"],
                subset["density"],
                linestyle=style,
                color=color,
                linewidth=LINEWIDTH,
                label=f"{dname} ({label})"
            )
    if X_RANGE is not None:
        plt.xlim(X_RANGE)
    plt.xlabel("Min-K++ Score", fontsize=FONTSIZE_LABEL)
    plt.ylabel("Density", fontsize=FONTSIZE_LABEL)
    plt.title(f"Min-K++ KDE\nModel: {model_name}", fontsize=FONTSIZE_TITLE)

    plt.legend(fontsize=FONTSIZE_LEGEND, ncol=2)
    plt.tight_layout()

    plt.savefig(f"{OUT_DIR}/minkpp_kde_{model_name}.png", dpi=500)
    plt.savefig(f"{OUT_DIR}/minkpp_kde_{model_name}.pdf", bbox_inches="tight")
    plt.close()


# ================= RUN =================
if __name__ == "__main__":
    for model in MODELS:
        print(f" === Plotting for {model} ===")
        plot_kde(model)
        # plot_histogram(model)

    print("\n === All plots generated ===")