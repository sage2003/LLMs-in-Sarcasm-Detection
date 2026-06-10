import os
import random
import pandas as pd
from tqdm import tqdm
from groq import Groq
import time
import re

# ----------------------------------------------------
# Setup
# ----------------------------------------------------
os.environ["GROQ_API_KEY"] = ""   # <-- replace this
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL_NAME = "llama-3.3-70b-versatile"

# ----------------------------------------------------
# Load Natural Dataset (SARC V1)
# ----------------------------------------------------
DATASET_PATH = "Datasets/SARC-V1/SARC-V1-shuffled.csv"
df = pd.read_csv(DATASET_PATH)

# Normalize columns
if "sentence" in df.columns:
    text_col = "sentence"
elif "text" in df.columns:
    text_col = "text"
else:
    raise ValueError("No text column found")

if "label" in df.columns:
    label_col = "label"
elif "class" in df.columns:
    label_col = "class"
else:
    raise ValueError("No label column found")

df = df[[text_col, label_col]].dropna()
df.columns = ["sentence", "label"]

# Normalize labels
label_map = {
    1: "sarcastic",
    0: "not_sarcastic",
    "sarc": "sarcastic",
    "notsarc": "not_sarcastic",
    "sarcastic": "sarcastic",
    "not_sarcastic": "not_sarcastic"
}
df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"])

# ----------------------------------------------------
# Split by Class (for conditioning)
# ----------------------------------------------------
sarc_df = df[df["label"] == "sarcastic"].reset_index(drop=True)
nonsarc_df = df[df["label"] == "not_sarcastic"].reset_index(drop=True)

# ----------------------------------------------------
# Few-shot Prompt Builder
# ----------------------------------------------------
def build_fewshot_prompt(examples, label):
    tone = "sarcastic" if label == "sarcastic" else "sincere and non-sarcastic"

    example_block = "\n".join(
        [f'{i+1}. "{ex}"' for i, ex in enumerate(examples)]
    )


    prompt_a = (
        f"You are generating English sentences written in a casual, conversational tone.\n\n"
        f"Below are examples of {tone} sentences written by humans. "
        f"They are natural and context-dependent.\n\n"
        f"Examples:\n{example_block}\n\n"
        f"Now generate ONE NEW {tone} sentence that matches the style "
        f"and subtlety of the examples above.\n"
        f"Use natural conversational phrasing, not formal or essay-like language.\n"
        f"Do NOT paraphrase or copy the examples.\n"
        f"Output only the sentence."
    )

    prompt_b = (
        f"You are generating short, direct English sentences.\n\n"
        f"Below are examples of {tone} sentences written by humans. "
        f"They are natural and context-dependent.\n\n"
        f"Examples:\n{example_block}\n\n"
        f"Now generate ONE NEW {tone} sentence that matches the style "
        f"and subtlety of the examples above.\n"
        f"Keep the sentence concise and deadpan; avoid explanations or hedging.\n"
        f"Do NOT paraphrase or copy the examples.\n"
        f"Output only the sentence."
    )

    prompt_c = (
        f"You are generating English sentences that describe a specific situation or moment.\n\n"
        f"Below are examples of {tone} sentences written by humans. "
        f"They are natural and context-dependent.\n\n"
        f"Examples:\n{example_block}\n\n"
        f"Now generate ONE NEW {tone} sentence that matches the style "
        f"and subtlety of the examples above.\n"
        f"Focus on a concrete situation rather than abstract argument.\n"
        f"Do NOT paraphrase or copy the examples.\n"
        f"Output only the sentence."
    )   

    prompt_templates = [prompt_a, prompt_b, prompt_c]
    prompt = random.choice(prompt_templates)

    # prompt = (
    #     f"You are generating English sentences.\n\n"
    #     f"Below are examples of {tone} sentences written by humans. "
    #     f"They are natural and context-dependent.\n\n"
    #     f"Examples:\n{example_block}\n\n"
    #     f"Now generate ONE NEW {tone} sentence that matches the style "
    #     f"and subtlety of the examples above.\n"
    #     f"Do NOT paraphrase or copy the examples.\n"
    #     f"Output only the sentence."
    # )

    return prompt

# ----------------------------------------------------
#  Model Call
# ----------------------------------------------------
def generate_sentence(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": (
                "You are a text generation engine. "
                "Produce only the requested sentence. "
                "No explanations or extra text."
            )},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=60,
        top_p=0.95,
    )
    time.sleep(2)
    return response.choices[0].message.content.strip()

# ----------------------------------------------------
#  Post-generation quality filters
# ----------------------------------------------------
BAD_TOKENS = [
    "EntityState", "PROT", "asu", "872", "inf",
    "casting", "couchasu", "Decomastle"
]

def is_valid_sentence(s: str) -> bool:
    # Basic length check
    if not isinstance(s, str):
        return False

    words = s.split()
    if len(words) < 6:
        return False

    # Garbage token check
    for tok in BAD_TOKENS:
        if tok.lower() in s.lower():
            return False

    # Excessive non-alphanumeric symbols
    non_alnum = re.findall(r"[^a-zA-Z0-9\s.,'\"!?]", s)
    if len(non_alnum) > 5:
        return False

    # Repeated word spam (e.g., "rapid rapid rapid")
    for w in set(words):
        if words.count(w) > 4:
            return False

    return True

# ----------------------------------------------------
#  Dataset Generation
# ----------------------------------------------------
def generate_intermediate_dataset(
    num_samples=300,
    shots_per_prompt=3,
    max_attempts_per_sample=3
):
    data = []

    for _ in tqdm(range(num_samples)):
        label = random.choice(["sarcastic", "not_sarcastic"])
        source_df = sarc_df if label == "sarcastic" else nonsarc_df

        examples = source_df.sample(
            shots_per_prompt,
            random_state=random.randint(0, 1_000_000)
        )["sentence"].tolist()

        prompt = build_fewshot_prompt(examples, label)

        attempts = 0
        while attempts < max_attempts_per_sample:
            try:
                sentence = generate_sentence(prompt)
                attempts += 1

                if is_valid_sentence(sentence):
                    data.append({
                        "sentence": sentence,
                        "label": label
                    })
                    break  # accept sentence

            except Exception as e:
                print("Error:", e)
                break

    return pd.DataFrame(data)

# ----------------------------------------------------
#  Run Generation
# ----------------------------------------------------
intermediate_df = generate_intermediate_dataset(
    num_samples=200,
    shots_per_prompt=3
)

intermediate_df = intermediate_df.sample(frac=1).reset_index(drop=True)
intermediate_df.to_csv(
    "anchored_fewshot_sarc_llama70b.csv",
    index=False
)

print(f"=== Generated {len(intermediate_df)} anchored synthetic samples ===")
