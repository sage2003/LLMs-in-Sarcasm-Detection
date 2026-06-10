import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
data = {
    "Llama-3.1-8B": {
        "SARC-V1 (Natural)": [0.606, 0.578, 0.950, 0.719],
        "Fewshot-llama-70b \n (Natural-Generated)": [0.904, 0.846, 0.998, 0.915],
        "Kimi-k2 (Synthetic)": [0.961, 0.932, 1.000, 0.964],
    },
    "Qwen2.5-14B": {
        "SARC-V1 (Natural)": [0.669, 0.677, 0.717, 0.696],
        "Fewshot-llama-70b \n (Natural-Generated)": [0.973, 0.954, 0.996, 0.974],
        "Kimi-k2 (Synthetic)": [0.992, 0.985, 1.000, 0.992],
    },
    "Gemma-2-9B-it": {
        "SARC-V1 (Natural)": [0.560, 0.547, 0.983, 0.703],
        "Fewshot-llama-70b \n (Natural-Generated)": [0.848, 0.787, 0.998, 0.880],
        "Kimi-k2 (Synthetic)": [0.938, 0.895, 1.000, 0.945],
    },
    "Phi-3-medium-4k": {
        "SARC-V1 (Natural)": [0.614, 0.586, 0.927, 0.718],
        "Fewshot-llama-70b \n (Natural-Generated)": [0.881, 0.814, 1.000, 0.897],
        "Kimi-k2 (Synthetic)": [0.973, 0.951, 1.000, 0.975],
    },
}

metrics = ["Accuracy", "Precision", "Recall", "F1"]

# -----------------------------
# FONT CONFIG
# -----------------------------
FONTSIZE_BASE = 16
FONTSIZE_TITLE = 22
FONTSIZE_LABEL = 20
FONTSIZE_TICK = 15
FONTSIZE_LEGEND = 20

# -----------------------------
# Global style (paper-safe)
# -----------------------------
plt.rcParams.update({
    "font.size": FONTSIZE_BASE,
    "axes.titlesize": FONTSIZE_TITLE,
    "axes.labelsize": FONTSIZE_LABEL,
    "xtick.labelsize": FONTSIZE_TICK,
    "ytick.labelsize": FONTSIZE_TICK,
    "legend.fontsize": FONTSIZE_LEGEND,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -----------------------------
# Plotting
# -----------------------------
for model, datasets in data.items():
    dataset_names = list(datasets.keys())
    values = np.array(list(datasets.values()))

    x = np.arange(len(dataset_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            values[:, i],
            width,
            label=metric,
            alpha=0.85
        )

    # Axes
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(dataset_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=FONTSIZE_LABEL)
    ax.set_title(f"{model}: Performance Across Datasets", pad=12, fontsize=FONTSIZE_TITLE)

    ax.tick_params(axis="x", pad=6, labelsize=FONTSIZE_TICK)
    ax.tick_params(axis="y", labelsize=FONTSIZE_TICK)

    # Light grid
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    # Legend below plot
    ax.legend(
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        fontsize=FONTSIZE_LEGEND
    )

    # Reserve space for legend
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    plt.savefig(
        f"Final_Plots/Phase-1/n_{model}_metrics_comparison.pdf",
        dpi=500,
        bbox_inches="tight"
    )
    plt.show()