import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import gc
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# --- 1. Setup & Configuration ---

# NOTE: Even in 4-bit, a 14B model requires ~10-12GB VRAM minimum.
BATCH_SIZE = 5  

OUTPUT_DIR = "Outputs/Phase-2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_DIR = "../CSV/Phase-2"
os.makedirs(CSV_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Models registry
MODELS = {
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "phi-3-medium-4k-instruct": "microsoft/Phi-3-medium-4k-instruct",    
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct"
}
    

# Dataset registry
DATASETS = {
    "Llama70B_synthetic": "../Datasets/Synthetic/Llama-3.3-70b-versatile/sarcasm_dataset_llama3.3-70b.csv",
    "KimiK2_synthetic": "../Datasets/Synthetic/kimi-k2-instruct-0905/sarcasm_dataset_kimi-k2-instruct-0905.csv",
    "SARC-V2_HYP": "../Datasets/SARC-V2/HYP-sarc-notsarc.csv",
    "SARC-V1": "../Datasets/SARC-V1/SARC-V1.csv",
    "SARC-V2_GEN": "../Datasets/SARC-V2/GEN-sarc-notsarc.csv",
    "SARC-V2_RQ": "../Datasets/SARC-V2/RQ-sarc-notsarc.csv",
    "News_Headlines": "../Datasets/News_Headlines/Sarcasm_Headlines_Dataset.csv",
    "Natural-Generated": "../Datasets/Intermediate/anchored_fewshot_sarc_llama70b.csv"
}

# --- 2. Embedding Extraction Function ---

def get_mean_pooled_embeddings(batch_sentences, model, tokenizer, layer_index):
    inputs = tokenizer(
        batch_sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer_index]
    attention_mask = inputs['attention_mask']

    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

    return (sum_embeddings / sum_mask).detach().cpu().numpy()

# --- 3. Main Automation Loop ---

for dataset_name, dataset_path in DATASETS.items():

    if not os.path.exists(dataset_path):
        print(f"Dataset not found, skipping: {dataset_path}")
        continue

    print(f"\n === Loading dataset: {dataset_name} ===")
    df = pd.read_csv(dataset_path)

    # Dataset normalization
    if "text" in df.columns:
        text_col = "text"
    elif "sentence" in df.columns:
        text_col = "sentence"
    elif "headline" in df.columns:
        text_col = "headline"
    else:
        print(" === No valid text column found, skipping. ===")
        continue

    label_col = "class" if "class" in df.columns else "label"
    df = df[[text_col, label_col]].dropna()

    sentences = df[text_col].tolist()
    labels = df[label_col].tolist()
    # --- Label normalization ---
    labels = [
        "sarcastic" if str(l).lower() in ["sarc", "sarcastic", "1"]
        else "not_sarcastic"
        for l in labels
    ]

    for model_tag, model_id in MODELS.items():

        print(f"\n === Loading model: {model_tag} ===")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        model.eval()

        # Layer selection
        dummy = tokenizer("test", return_tensors="pt").to(model.device)
        with torch.no_grad():
            hs = model(**dummy, output_hidden_states=True).hidden_states
        layer_map = {
            "first": 0,
            "middle": len(hs)//2,
            "last": -1
        }   
        

        for layer_name, layer_idx in layer_map.items():

            print(f" Extracting embeddings | Layer: {layer_name}")

            all_embeddings = []

            # for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
            #     batch = sentences[i:i+BATCH_SIZE]
            #     emb = get_mean_pooled_embeddings(batch, model, tokenizer, layer_idx)
            #     all_embeddings.append(emb)

            #     if i % 10 == 0:
            #         gc.collect()
            #         torch.cuda.empty_cache()

            # all_embeddings = np.concatenate(all_embeddings).astype(np.float32)

            for i in tqdm(range(0, len(sentences), BATCH_SIZE)):
                batch = sentences[i : i + BATCH_SIZE]

                try:
                    batch_embeddings = get_mean_pooled_embeddings(
                        batch, model, tokenizer, layer_idx
                    )
                    all_embeddings.append(batch_embeddings)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\nOOM at index {i}. Try lowering BATCH_SIZE.")
                        torch.cuda.empty_cache()
                        raise e
                    else:
                        raise e

                # Aggressive cleanup after EVERY batch for small GPUs
                if i % 10 == 0:  # Run every 10 batches to avoid slowing down too much
                    gc.collect()
                    torch.cuda.empty_cache()

            # IMPORTANT: keep axis=0 (do NOT remove this)
            all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

            # --- Dimensionality Reduction ---
            print("Running t-SNE...")
            perp = min(30, len(all_embeddings)-1)
            tsne_2d = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(all_embeddings)
            tsne_3d = TSNE(n_components=3, perplexity=perp, random_state=42).fit_transform(all_embeddings)

            print("Running UMAP...")
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
            umap_2d = reducer.fit_transform(all_embeddings)
            umap_3d = umap.UMAP(n_components=3, metric='cosine', random_state=42).fit_transform(all_embeddings)

            # --- Visualization ---

            print("Generating plots...")
            df_plot = pd.DataFrame({
                "model":model_tag,
                "dataset": dataset_name,
                "layer": layer_name,
                "label": labels,
                "tsne-1": tsne_2d[:,0], "tsne-2": tsne_2d[:,1], "tsne-3": tsne_3d[:, 2],
                "umap-1": umap_2d[:,0], "umap-2": umap_2d[:,1], "umap-3": umap_3d[:, 2]
            })


            # -------- CSV: embedding coordinates (final plotted values) --------
            csv_name = f"{CSV_DIR}/embedding_{model_tag}_{dataset_name}_{layer_name}.csv"
            df_plot.to_csv(csv_name, index=False)
            print(f" === Saved embedding CSV: {csv_name} ===")

            fname_base = f"{model_tag}_{dataset_name}_{layer_name}"

            palette = {"sarcastic": "tab:red", "not_sarcastic": "tab:blue"}

            fig = plt.figure(figsize=(16, 6))

            # --- t-SNE 2D ---
            ax1 = fig.add_subplot(1, 2, 1)
            sns.scatterplot(
                data=df_plot,
                x="tsne-1",
                y="tsne-2",
                hue="label",
                palette=palette,
                alpha=0.6,
                ax=ax1,
                legend=True
            )
            ax1.set_title(f"t-SNE 2D | {fname_base}")

            # --- UMAP 2D ---
            ax2 = fig.add_subplot(1, 2, 2)
            sns.scatterplot(
                data=df_plot,
                x="umap-1",
                y="umap-2",
                hue="label",
                palette=palette,
                alpha=0.6,
                ax=ax2,
                legend=False  # avoid duplicate legend
            )
            ax2.set_title(f"UMAP 2D | {fname_base}")

            plt.suptitle(
                f"2D Embedding Visualization - {model_tag} - {dataset_name} - {layer_name}",
                fontsize=14,
                y=0.98
            )

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)

            plt.savefig(f"{OUTPUT_DIR}/embedding_2d_{fname_base}.png", dpi=300)
            plt.close()

            fig = plt.figure(figsize=(16, 8))

            # --- t-SNE 3D ---
            ax1 = fig.add_subplot(1, 2, 1, projection="3d")
            for lbl, color in palette.items():
                subset = df_plot[df_plot["label"] == lbl]
                ax1.scatter(
                    subset["tsne-1"],
                    subset["tsne-2"],
                    subset["tsne-3"],
                    label=lbl,
                    c=color,
                    s=8,
                    alpha=0.5
                )

            ax1.set_title(f"t-SNE 3D | {fname_base}")
            ax1.set_xlabel("t-SNE-1")
            ax1.set_ylabel("t-SNE-2")
            ax1.set_zlabel("t-SNE-3")
            ax1.legend()

            # --- UMAP 3D ---
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            for lbl, color in palette.items():
                subset = df_plot[df_plot["label"] == lbl]
                ax2.scatter(
                    subset["umap-1"],
                    subset["umap-2"],
                    subset["umap-3"],
                    label=lbl,
                    c=color,
                    s=8,
                    alpha=0.5
                )

            ax2.set_title(f"UMAP 3D | {fname_base}")
            ax2.set_xlabel("UMAP-1")
            ax2.set_ylabel("UMAP-2")
            ax2.set_zlabel("UMAP-3")
            ax2.legend()

            plt.suptitle(
                f"3D Embedding Visualization - {model_tag} - {dataset_name} - {layer_name}",
                fontsize=14,
                y=0.98
            )

            plt.tight_layout()
            plt.subplots_adjust(top=0.88)

            plt.savefig(f"{OUTPUT_DIR}/embedding_3d_{fname_base}.png", dpi=300)
            plt.close()

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded to free memory for visualization.")
        

print("\n === Phase-2 automation completed. ===")
