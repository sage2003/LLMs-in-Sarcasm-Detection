import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import gaussian_kde

# ================= USER CONFIG =================
CSV_DIR = "../CSV/Phase-2"
OUT_DIR = "Final_Plots/Embedding"

FIGSIZE = (16, 7)
FONTSIZE_LABEL = 16
FONTSIZE_TITLE = 24
FONTSIZE_LEGEND = 14

POINT_SIZE = 20          # CRITICAL for sharpness
ALPHA = 0.6             # works well with density sorting
DPI = 500
# ==============================================

os.makedirs(OUT_DIR, exist_ok=True)

palette = {
    "sarcastic": "tab:red",
    "not_sarcastic": "tab:blue"
}

sns.set_style("white")


# -------- Density-aware scatter --------
def density_scatter(ax, x, y, labels, title):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()  # low density first, high density last

    x, y, z, labels = x[idx], y[idx], z[idx], labels[idx]

    colors = [palette[l] for l in labels]

    ax.scatter(
        x,
        y,
        c=colors,
        s=POINT_SIZE,
        alpha=ALPHA,
        edgecolors="none"
    )

    ax.set_title(title, fontsize=FONTSIZE_TITLE)
    ax.set_xlabel(title.split()[0] + "-1", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel(title.split()[0] + "-2", fontsize=FONTSIZE_LABEL)


def plot_2d_embedding(csv_path):
    df = pd.read_csv(csv_path)

    # -------- metadata --------
    model = df["model"].iloc[0]
    dataset = df["dataset"].iloc[0]
    layer = df["layer"].iloc[0]

    fname_base = f"{model}_{dataset}_{layer}"

    # -------- data --------
    labels = df["label"].values

    # -------- plot --------
    fig = plt.figure(figsize=FIGSIZE)

    # --- t-SNE ---
    ax1 = fig.add_subplot(1, 2, 1)
    density_scatter(
        ax1,
        df["tsne-1"].values,
        df["tsne-2"].values,
        labels,
        "t-SNE 2D"
    )

    # --- UMAP ---
    ax2 = fig.add_subplot(1, 2, 2)
    density_scatter(
        ax2,
        df["umap-1"].values,
        df["umap-2"].values,
        labels,
        "UMAP 2D"
    )

    # -------- legend (only once) --------
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='sarcastic',
                   markerfacecolor='tab:red', markersize=6),
        plt.Line2D([0], [0], marker='o', color='w', label='not_sarcastic',
                   markerfacecolor='tab:blue', markersize=6)
    ]
    ax1.legend(handles=handles, fontsize=FONTSIZE_LEGEND)

    # -------- title --------
    plt.suptitle(
        f"2D Embedding Visualization\nModel: {model} | Dataset: {dataset} | Layer: {layer}",
        fontsize=FONTSIZE_TITLE
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.80)

    # -------- save --------
    save_name = f"embedding_2d_{fname_base}"
    plt.savefig(f"{OUT_DIR}/{save_name}.png", dpi=DPI)
    plt.savefig(f"{OUT_DIR}/{save_name}.pdf", bbox_inches="tight")
    plt.close()


# ================= RUN =================
if __name__ == "__main__":

    for file in os.listdir(CSV_DIR):
        if file.startswith("embedding_") and file.endswith(".csv"):
            csv_path = os.path.join(CSV_DIR, file)

            print(f"📊 Plotting {file}")
            plot_2d_embedding(csv_path)

    print("\n === All embedding plots generated ===")