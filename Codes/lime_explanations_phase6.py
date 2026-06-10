# phase6_lime_multimodel_final_with_tqdm.py

import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from lime.lime_text import LimeTextExplainer
import pandas as pd
from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================

MODELS = {
    "phi-3-medium": "microsoft/Phi-3-medium-4k-instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

NUM_SAMPLES = 2000
OUT_DIR = "lime_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================
# PROMPT FACTORY
# ======================================================

def build_prompt(model_tag: str, sentence: str) -> str:

    if model_tag == "qwen2.5-14b":
        return f"""You are known for being able to precisely classify whether a sentence is sarcastic or not.

Sentence: "{sentence}"

Is the sentence sarcastic? Answer strictly with only "Yes" or "No":"""

    elif model_tag == "llama3.1-8b":
        return f"""<|start_header_id|>system<|end_header_id|>

Your task is to classify if a sentence is sarcastic. Sarcasm often involves irony, exaggeration, or mockery.

Answer with only "Yes" or "No".

Sentence: {sentence}<|eot_id|><|start_header_id|>user<|end_header_id|>

Is the response sarcastic?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    elif model_tag == "phi-3-medium":
        return f"""Is the following sentence sarcastic?
Answer Yes or No only.

Sentence: "{sentence}"

Answer:"""

    elif model_tag == "gemma-2-9b":
        return f"""Answer Yes or No only.

Rules:
- Determine whether the sentence is sarcastic.
- Answer with exactly one word: Yes or No.
- If you are unsure, answer No.

Sentence: "{sentence}"

Answer:"""

    else:
        raise ValueError(f"Unknown model tag: {model_tag}")

# ======================================================
# LIME PREDICTOR
# ======================================================

def make_predictor(model, tokenizer, model_tag):

    def predictor(texts):
        probs = []

        for text in texts:
            prompt = build_prompt(model_tag, text)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            scores = out.scores[0][0]
            yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_id = tokenizer.encode("No", add_special_tokens=False)[0]

            p = torch.softmax(
                torch.tensor([scores[no_id], scores[yes_id]]),
                dim=0
            )
            probs.append(p.cpu().numpy())

        return np.array(probs)

    return predictor

# ======================================================
# MAIN LOOP (WITH tqdm)
# ======================================================

def load_interpretability_sentences(path="interpretability_sentences.csv"):
        df = pd.read_csv(path)
        out = {}
        for _, r in df.iterrows():
            out.setdefault(r["dataset"], {})[r["label"]] = r["sentence"]
        return out

data = load_interpretability_sentences()

for model_tag, model_name in tqdm(MODELS.items(), desc="Models", position=0):

    print(f"\n=== Running LIME for {model_tag} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16
    ).eval()

    predictor = make_predictor(model, tokenizer, model_tag)

    explainer = LimeTextExplainer(
        class_names=["Not Sarcastic", "Sarcastic"]
    )

    # flatten (dataset, label) pairs for clean tqdm
    pairs = [
        (dname, label)
        for dname in data
        for label in data[dname]
    ]

    for dname, label in tqdm(
        pairs,
        desc=f"LIME ({model_tag})",
        position=1,
        leave=False
    ):
        sentence = data[dname][label]

        exp = explainer.explain_instance(
            sentence,
            predictor,
            num_samples=NUM_SAMPLES
        )

        fname = f"{model_tag}_{dname}_{label}.html"
        exp.save_to_file(os.path.join(OUT_DIR, fname))

    del model
    torch.cuda.empty_cache()

print("\n=== Phase-6 LIME completed successfully ===")
