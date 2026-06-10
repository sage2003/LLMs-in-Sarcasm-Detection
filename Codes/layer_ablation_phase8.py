import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import gc

# ======================================================
# CONFIG
# ======================================================

MODELS = {
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
}

EVAL_CSV = "Datasets/SARC-V1/SARC-V1-ablation_fixed.csv"  # 50–100 rows
OUT_DIR = "ablation_outputs"
DTYPE = torch.float16

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# PROMPTS (FROM PHASE 6)
# ======================================================

def build_prompt(model_tag, sentence):
    if model_tag == "qwen2.5-14b":
        return f"""You are known for being able to precisely classify whether a sentence is sarcastic or not.

Sentence: "{sentence}"

Is the sentence sarcastic? Answer strictly with only "Yes" or "No":"""

    elif model_tag == "llama3.1-8b":
        return f"""<|start_header_id|>system<|end_header_id|>
Your task is to classify if a sentence is sarcastic.
Answer with only "Yes" or "No".

Sentence: {sentence}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    elif model_tag == "phi-3-medium":
        return f"""Is the following sentence sarcastic?
Answer Yes or No only.

Sentence: "{sentence}"

Answer:"""

    elif model_tag == "gemma-2-9b":
        return f"""Answer Yes or No only.

Sentence: "{sentence}"

Answer:"""

# ======================================================
# IDENTITY WRAPPER (ARCHITECTURE SAFE)
# ======================================================

class IdentityWrapperLayer(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.original_layer = original_layer
        for attr in dir(original_layer):
            if not attr.startswith("_") and not hasattr(self, attr):
                try:
                    setattr(self, attr, getattr(original_layer, attr))
                except Exception:
                    pass

    def forward(self, *args, **kwargs):
        outputs = self.original_layer(*args, **kwargs)
        if isinstance(outputs, tuple):
            return (args[0],) + outputs[1:]
        return args[0]

# ======================================================
# BLOCK ACCESS
# ======================================================

def get_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers          # LLaMA / Gemma
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h         # Qwen / Phi
    raise RuntimeError("Unsupported architecture")

# ======================================================
# PREDICTION (MEMORY SAFE)
# ======================================================

@torch.no_grad()
def predict(sentence, model, tokenizer, model_tag):
    prompt = build_prompt(model_tag, sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        use_cache=False,     
    )

    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(gen, skip_special_tokens=True).lower()
    return 1 if "yes" in answer else 0

# ======================================================
# CHUNK-WISE ABLATION
# ======================================================

def run_chunk_ablation(model, tokenizer, model_tag, sentences, labels, blocks):
    L = len(blocks)
    chunks = {
        "Early": range(0, L // 4),
        "Early-Middle": range(L // 4, L // 2),
        "Late-Middle": range(L // 2, 3 * L // 4),
        "Late": range(3 * L // 4, L),
    }

    with torch.inference_mode():
        baseline_preds = [predict(s, model, tokenizer, model_tag) for s in sentences]
    baseline_acc = accuracy_score(labels, baseline_preds)

    drops = {}
    for name, idxs in chunks.items():
        backup = {i: blocks[i] for i in idxs}
        for i in idxs:
            blocks[i] = IdentityWrapperLayer(blocks[i])

        with torch.inference_mode():
            preds = [predict(s, model, tokenizer, model_tag) for s in sentences]
        drops[name] = baseline_acc - accuracy_score(labels, preds)

        for i, blk in backup.items():
            blocks[i] = blk

        torch.cuda.empty_cache()
        gc.collect()

    return drops

# ======================================================
# MAIN
# ======================================================

df = pd.read_csv(EVAL_CSV)
sentences = df["text"].tolist()
labels = [1 if c.lower() == "sarc" else 0 for c in df["class"]]

for model_tag, model_name in MODELS.items():

    print(f"\n=== Phase 8 | {model_tag} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=DTYPE,
        quantization_config=bnb_cfg
    ).eval()

    blocks = get_blocks(model)
    num_layers = len(blocks)

    # ---------- Baseline ----------
    with torch.inference_mode():
        baseline_preds = [predict(s, model, tokenizer, model_tag) for s in sentences]
    baseline_acc = accuracy_score(labels, baseline_preds)

    # ---------- Layer-wise ----------
    drops = []
    for i in tqdm(range(num_layers), desc="Layer ablation"):
        original = blocks[i]
        blocks[i] = IdentityWrapperLayer(original)

        with torch.inference_mode():
            preds = [predict(s, model, tokenizer, model_tag) for s in sentences]
        drops.append(baseline_acc - accuracy_score(labels, preds))

        blocks[i] = original
        torch.cuda.empty_cache()
        gc.collect()

    # Save CSV
    pd.Series(drops).to_csv(f"{OUT_DIR}/layerwise_{model_tag}.csv")

    # Plot layer-wise
    plt.figure(figsize=(14, 5))
    plt.bar(range(num_layers), drops)
    plt.axhline(0, color="black")
    plt.xticks(range(num_layers), [f"L{i:02d}" for i in range(num_layers)], rotation=90)
    plt.ylabel("Accuracy drop")
    plt.title(f"Layer-wise Block Removal\nModel: {model_tag} | Dataset: SARC-V1")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/layerwise_{model_tag}.png", dpi=300)
    plt.close()

    # ---------- Chunk-wise ----------
    chunk_drops = run_chunk_ablation(
        model, tokenizer, model_tag, sentences, labels, blocks
    )

    pd.Series(chunk_drops).to_csv(f"{OUT_DIR}/chunkwise_{model_tag}.csv")

    plt.figure(figsize=(6, 4))
    plt.bar(chunk_drops.keys(), chunk_drops.values())
    plt.axhline(0, color="black")
    plt.ylabel("Accuracy drop")
    plt.title(f"Chunk-wise Block Removal\nModel: {model_tag} | Dataset: SARC-V1")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/chunkwise_{model_tag}.png", dpi=300)
    plt.close()

    del model
    torch.cuda.empty_cache()
    gc.collect()

print("\n=== Phase 8 complete ===")
