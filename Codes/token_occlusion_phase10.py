import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# ======================================================
# CONFIG
# ======================================================

MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
}

DTYPE = torch.float16
DEVICE = "cuda"

SENTENCE_CSV = "interpretability_sentences.csv"
OUT_DIR = "second_phase10_token_occlusion"

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# PROMPT FACTORY (FROM PHASE 6)
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

    else:
        raise ValueError(model_tag)

# ======================================================
# SAFE SARCASTIC PROBABILITY EXTRACTION
# ======================================================

@torch.no_grad()
def sarcasm_probability(sentence, model, tokenizer, model_tag):
    prompt = build_prompt(model_tag, sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    logits = outputs.scores[0][0]

    # robust multi-token handling
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids  = tokenizer.encode("No",  add_special_tokens=False)

    yes_score = logits[yes_ids].sum()
    no_score  = logits[no_ids].sum()

    probs = torch.softmax(torch.stack([yes_score, no_score]), dim=0)
    return probs[0].item()   # P(sarcastic)

# ======================================================
# WORD-LEVEL OCCLUSION
# ======================================================

def run_word_occlusion(sentence, model, tokenizer, model_tag):
    words = sentence.split()
    base_prob = sarcasm_probability(sentence, model, tokenizer, model_tag)

    impacts = []

    for i in range(len(words)):
        occluded_words = words[:i] + words[i+1:]
        occluded_sentence = " ".join(occluded_words)

        new_prob = sarcasm_probability(
            occluded_sentence, model, tokenizer, model_tag
        )

        impacts.append(base_prob - new_prob)

    return words, impacts, base_prob

# ======================================================
# MAIN
# ======================================================

df = pd.read_csv(SENTENCE_CSV)
# Expected columns:
# dataset | label | sentence
# label ∈ {"sarcastic", "not_sarcastic"}

for model_tag, model_name in MODELS.items():

    print(f"\n=== Phase 10 | Model: {model_tag} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=DTYPE,
        low_cpu_mem_usage=True
    ).eval()

    for _, row in df.iterrows():

        dataset = row["dataset"]
        label = row["label"]
        sentence = row["sentence"]

        words, impacts, base_prob = run_word_occlusion(
            sentence, model, tokenizer, model_tag
        )

        # ---------- Plot ----------
        plt.figure(figsize=(max(6, 0.8 * len(words)), 3))

        V = 0.4

        norm = mcolors.TwoSlopeNorm(
            vmin=-V,
            vcenter=0.0,
            vmax=V
        )


        sns.heatmap(
            [impacts],
            xticklabels=words,
            yticklabels=["Impact"],
            cmap="coolwarm",
            norm=norm,
            annot=True,
            fmt=".3f"
        )       

        plt.title(
            f"Word Occlusion Attribution\n"
            f"Model: {model_tag} | Dataset: {dataset} | Label: {label}\n"
            f"P(sarcastic) = {base_prob:.3f}"
        )

        plt.tight_layout()

        fname = (
            f"occlusion_{model_tag}_"
            f"{dataset}_{label}.png"
        )

        plt.savefig(
            os.path.join(OUT_DIR, fname),
            dpi=500
        )
        plt.close()

    del model
    torch.cuda.empty_cache()

print("\n=== Phase 10 completed successfully ===")
