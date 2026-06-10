import os
import time
import random
import pandas as pd
from tqdm import tqdm
from groq import Groq

# ====================================================
# CONFIG
# ====================================================

os.environ["GROQ_API_KEY"] = ""
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL_NAME = "llama-3.3-70b-versatile"
# MODEL_NAME = "moonshotai/kimi-k2-instruct-0905"

TOTAL_SAMPLES = 300
SLEEP_TIME = 2.0

OUTPUT_FILE = "sarcasm_dataset_notopic_random_llama70b.csv"

# ====================================================
# PROMPTS
# ====================================================

def generate_prompt(sarcastic: bool) -> str:
    if sarcastic:
        return (
            "Write exactly ONE English piece of text that is sarcastic.\n"
            "The piece of text can be about anything.\n"
            "Avoid common sarcastic clichés such as "
            "\"oh great\", \"said no one ever\", or traffic jams.\n"
            "Try to be original and avoid repeating common patterns.\n"
            "The length should be between 5 and 50 words.\n"
            "Output only the text."
        )
    else:
        return (
            "Write exactly ONE English piece of text that is sincere and not sarcastic.\n"
            "The piece of text can be about anything.\n"
            "Avoid generic gratitude or nature-related clichés.\n"
            "Try to be original and vary phrasing.\n"
            "The length should be between 5 and 50 words.\n"
            "Output only the text."
        )

# ====================================================
# MODEL CALL
# ====================================================

def call_model(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a text completion engine. "
                    "Produce only the requested text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        top_p=0.95,
        max_tokens=60,
    )
    time.sleep(SLEEP_TIME)
    return response.choices[0].message.content.strip()

# ====================================================
# DEDUP FILTER
# ====================================================

def normalize(text: str) -> str:
    return (
        text.lower()
        .replace(",", "")
        .replace(".", "")
        .replace("!", "")
        .replace("?", "")
        .replace('"', "")
        .strip()
    )

# ====================================================
# GENERATION
# ====================================================

def generate_dataset():
    data = []
    seen = set()

    with tqdm(total=TOTAL_SAMPLES) as pbar:
        while len(data) < TOTAL_SAMPLES:
            label = random.choice(["sarcastic", "not_sarcastic"])
            prompt = generate_prompt(label == "sarcastic")

            try:
                text = call_model(prompt)
            except Exception as e:
                print("API error:", e)
                continue

            words = text.split()
            if not (5 <= len(words) <= 50):
                continue

            norm = normalize(text)
            if norm in seen:
                continue

            seen.add(norm)
            data.append({"sentence": text, "label": label})
            pbar.update(1)

    return pd.DataFrame(data)

# ====================================================
# SAVE
# ====================================================

if __name__ == "__main__":
    df = generate_dataset()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n === Saved {len(df)} samples → {OUTPUT_FILE} ===")
