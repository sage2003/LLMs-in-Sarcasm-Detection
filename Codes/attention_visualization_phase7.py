# phase7_attention_multimodel_with_tqdm.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bertviz import head_view
from tqdm import tqdm
import pandas as pd

# ======================================================
# CONFIG
# ======================================================

MODELS = {
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
}

OUT_DIR = "Outputs/Phase-7_attention_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# LOAD FIXED SENTENCES
# ======================================================

def load_interpretability_sentences(path="interpretability_sentences.csv"):
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            out.setdefault(r["dataset"], {})[r["label"]] = r["sentence"]
        return out

data = load_interpretability_sentences()

# ======================================================
# MAIN LOOP WITH tqdm
# ======================================================

for model_tag, model_name in tqdm(MODELS.items(), desc="Models", position=0):

    print(f"\n=== Loading model: {model_tag} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        output_attentions=True
    ).eval()

    # Flatten (dataset, label) pairs for clean tqdm
    pairs = [
        (dname, label)
        for dname in data
        for label in data[dname]
    ]

    for dname, label in tqdm(
        pairs,
        desc=f"Attention ({model_tag})",
        position=1,
        leave=False
    ):
        sentence = data[dname][label]

        # Tokenize and run forward pass
        inputs = tokenizer.encode(sentence, return_tensors="pt")
        outputs = model(inputs)

        attentions = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(inputs[0])

        # Generate BertViz HTML
        html = head_view(attentions, tokens, html_action="return")

        fname = f"{model_tag}_{dname}_{label}.html"
        fpath = os.path.join(OUT_DIR, fname)

        with open(fpath, "w", encoding="utf-8") as f:
            f.write(html.data)

    del model
    torch.cuda.empty_cache()

print("\n=== Phase-7 attention visualization completed successfully ===")
