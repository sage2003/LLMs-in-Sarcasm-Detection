import os
import random
import pandas as pd
from tqdm import tqdm
from groq import Groq

import time

# ----------------------------------------------------
# Setup
# ----------------------------------------------------
# Set your Groq API key
os.environ["GROQ_API_KEY"] = ""   # <-- replace this
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Topics to diversify sarcasm across domains
# topics = [
#     "politics", "sports", "technology", "relationships", "social media", "education",
#     "daily life", "weather", "travel", "food", "movies", "music", "work", "college life",
#     "shopping", "fashion", "science", "health", "fitness", "gaming", "celebrities"
# ]

# topics = [
#     # Society & Institutions
#     "politics", "government policies", "bureaucracy", "public services",
#     "education", "college life", "academia", "grading systems",
#     "corporate culture", "office meetings", "human resources",

#     # Technology & Media
#     "technology", "artificial intelligence", "social media",
#     "online influencers", "tech support", "software updates",
#     "smartphones", "streaming platforms", "online ads",

#     # Daily Life & Frustrations
#     "daily life", "commuting", "traffic", "weather",
#     "customer service", "waiting in queues", "online shopping",
#     "delivery services", "home appliances",

#     # Work & Productivity
#     "work", "deadlines", "emails", "remote work",
#     "meetings that could be emails", "productivity tools",

#     # Relationships & Social Dynamics
#     "relationships", "dating", "breakups", "friendship",
#     "family gatherings", "group chats", "social expectations",

#     # Entertainment & Culture
#     "movies", "movie sequels", "music", "award shows",
#     "celebrity scandals", "reality TV", "fandoms",

#     # Health & Lifestyle
#     "health", "fitness", "dieting", "mental health",
#     "sleep schedules", "wellness trends",

#     # Science & Knowledge
#     "science", "research funding", "peer review",
#     "scientific publishing", "space exploration",
#     "climate change",

#     # Sports & Games
#     "sports", "refereeing decisions", "team loyalty",
#     "fantasy leagues", "e-sports", "gaming updates",

#     # Money & Consumerism
#     "shopping", "sales discounts", "subscriptions",
#     "pricing models", "hidden fees",

#     # Travel & Infrastructure
#     "travel", "flight delays", "airports",
#     "public transport", "tourist attractions"
# ]

# def generate_prompt(topic, sarcastic=True):
#     """Creates a strong instruction prompt for sarcasm generation."""
#     if sarcastic:
#         # return (
#         #     f"Write one sarcastic sentence about {topic}. "
#         #     f"Make the sarcasm witty and natural. "
#         #     f"Keep it human-like and contextually believable. "
#         #     f"Do not use hashtags or emojis. "
#         #     f"Avoid overly exaggerated sarcasm. "
#         #     f"Do not use phrases like 'I'm absolutely thrilled...' or 'Oh great!...'."
#         # )
#         return(
#             f"Write exactly ONE English sentence that is sarcastic.\n"
#             f"Topic: {topic}\n"
#             f"Sentence:"
#         )
#     # else:
#     #     return (
#     #         f"Write one sincere, non-sarcastic sentence about {topic}. "
#     #         f"It should sound genuine and positive or neutral."
#     #     )
#     else:
#         return(
#             f"Write exactly ONE English sentence that is sincere and not sarcastic.\n"
#         )


def generate_prompt(sarcastic=True):
    if sarcastic:
        return (
            "Write exactly ONE English piece of text that is sarcastic.\n"
            "The piece of text can be about anything.\n"
            "The length should be between 5 and 50 words.\n"
            "Output only the text."
        )
    else:
        return (
            "Write exactly ONE English piece of text that is sincere and not sarcastic.\n"
            "The piece of text can be about anything.\n"
            "The length should be between 5 and 50 words.\n"
            "Output only the text."
        )

def generate_response(prompt):
    """Calls Groq LLaMA-3 70B API for a single prompt."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # model="moonshotai/kimi-k2-instruct-0905",
        messages=[
            #{"role": "system", "content": "You are a precise sarcasm sentence generator."},    #Used for llama model
            {"role": "system", "content": (
                "You are a text completion engine. "
                "Your task is to fill in the requested text exactly. "
                "You must produce only the requested text, with no analysis, "
                "no explanation, and no extra content. "
                "Any text outside the requested output is invalid. "
            )},

            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=80,
        top_p=0.95,
    )
    time.sleep(2) # to avoid rate limits

    return response.choices[0].message.content.strip()

def generate_examples(num_examples=5000):
    """Generates sarcastic + non-sarcastic sentences and stores them."""
    data = []
    for _ in tqdm(range(num_examples)):
        # topic = random.choice(topics)
        sarcastic = random.choice([True, False])
        prompt = generate_prompt(sarcastic)

        try:
            sentence = generate_response(prompt)
            if len(sentence.split()) < 4:
                continue  # skip very short ones
            label = "sarcastic" if sarcastic else "not_sarcastic"
            data.append({"sentence": sentence, "label": label})
        except Exception as e:
            print("Error:", e)
            continue

    return pd.DataFrame(data)

# ----------------------------------------------------
# Generate Dataset
# ----------------------------------------------------
df = generate_examples(1000)

# Shuffle and reset index
df = df.sample(frac=1).reset_index(drop=True)
0
# Save to CSV
df.to_csv("sarcasm_dataset_llama3.3-70b.csv", index=False)
print(f"=== Done! Generated {len(df)} examples -> sarcasm_dataset_llama3.3-70b.csv ===")