import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= USER CONFIG =================
BASE_DIR = "../Outputs/Phase-8"
OUT_DIR = "Final_Plots/ablation"

# visual tweaks (same as before)
FIGSIZE = (12, 6)
FONTSIZE_LABEL = 16
FONTSIZE_TITLE = 22
FONTSIZE_TICKS = 12
# ==============================================

os.makedirs(OUT_DIR, exist_ok=True)


def plot_layerwise_ablation(csv_path, model_name, dataset_name):
    df = pd.read_csv(csv_path)

    # -------- detect columns safely --------
    if "layer" in df.columns:
        layer_col = "layer"
    else:
        layer_col = [c for c in df.columns if "layer" in c.lower()][0]

    if "0" in df.columns:
        drop_col = "0"
    else:
        drop_candidates = [c for c in df.columns if "drop" in c.lower()]
        drop_col = drop_candidates[0] if drop_candidates else df.columns[1]

    layers = df[layer_col].values
    drops = df[drop_col].values

    plt.figure(figsize=FIGSIZE)

    plt.bar(layers, drops)

    # zero baseline
    plt.axhline(0)

    plt.xlabel("Layer index", fontsize=FONTSIZE_LABEL)
    plt.ylabel("Accuracy drop", fontsize=FONTSIZE_LABEL)

    plt.title(
        f"Layer-wise Block Removal Ablation\nModel: {model_name} | Dataset: {dataset_name}",
        fontsize=FONTSIZE_TITLE
    )

    plt.xticks(
        layers,
        [f"L{int(l)}" for l in layers],
        rotation=90,
        fontsize=FONTSIZE_TICKS
    )

    plt.yticks(fontsize=FONTSIZE_TICKS)

    plt.tight_layout()

    # -------- Save --------
    save_name = f"{dataset_name}_layerwise_ablation_{model_name}"
    plt.savefig(f"{OUT_DIR}/{save_name}.png", dpi=500)
    plt.savefig(f"{OUT_DIR}/{save_name}.pdf", bbox_inches="tight")
    plt.close()


# ================= RUN =================
if __name__ == "__main__":

    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"\n=== Processing folder: {folder} ===")

        dataset_name = folder.replace("ablation_outputs_", "")

        for file in os.listdir(folder_path):
            if file.startswith("layerwise_drops_") and file.endswith(".csv"):

                model_name = file.replace("layerwise_drops_", "").replace(".csv", "")
                csv_path = os.path.join(folder_path, file)

                print(f" === Plotting {model_name} ({dataset_name}) ===")

                plot_layerwise_ablation(csv_path, model_name, dataset_name)

    print("\n === All ablation plots generated ===")