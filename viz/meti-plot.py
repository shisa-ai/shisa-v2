#!/usr/bin/env python3
# meti_plot.py
#
# Quick scatter plot of JA-vs-EN performance for a set of models.
# Uses a row-oriented data definition so column lengths can’t diverge.

import pandas as pd
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Data:  (license, model name, overall, ja, en)
# ──────────────────────────────────────────────────────────────────────────────
rows = [
    #  license      model-name                               overall  ja     en
    ("Apache 2.0",  "shisa-v2-qwen25-7b",                    62.96,  71.06, 54.86),
    ("Llama 3.1",   "shisa-v2-llama3.1-8b",                  62.79,  70.83, 54.75),
    ("Apache 2.0",  "shisa-v2-mistral-nemo-12b",             63.08,  72.83, 53.33),
    ("MIT",         "shisa-v2-unphi4-14b",                   67.99,  75.89, 60.10),
    ("Apache 2.0",  "shisa-v2-qwen2.5-32b",                  72.19,  76.97, 67.41),
    ("Llama 3.3",   "shisa-v2-llama-3.3-70b",                73.72,  79.72, 67.71),
    ("Llama 3.1",   "meti-geniac-405b-dpo.jury",             76.77,  81.32, 72.22),

    # Baselines / external models
    ("Llama 3.1",   "meta-llama/Llama-3.1-8B-Instruct",      53.65,  53.43, 53.88),
    ("Apache 2.0",  "Qwen/Qwen2.5-7B-Instruct",              61.71,  65.30, 58.11),
    ("Apache 2.0",  "mistralai/Mistral-Nemo-Instruct-2407",  53.26,  58.44, 48.07),
    ("MIT",         "microsoft/phi-4",                       66.80,  72.47, 61.14),
    ("Apache 2.0",  "Qwen/Qwen2.5-32B-Instruct",             66.73,  66.79, 66.67),
    ("Llama 3.3",   "meta-llama/Llama-3.3-70B-Instruct",     72.12,  72.75, 71.48),

    # OpenAI models
    ("Proprietary", "gpt-4.1-2025-04-14",                    83.74,  88.55, 78.94),
    ("Proprietary", "gpt-4o-2024-11-20",                     79.33,  85.32, 73.34),
    ("Proprietary", "gpt-4.1-mini-2025-04-14",               78.60,  84.63, 72.57),
    ("MIT",         "deepseek-ai/DeepSeek-V3-0324",          79.74,  82.95, 76.52),
    ("Proprietary", "gpt-4-turbo-2024-04-09",                72.57,  76.49, 68.64),
    ("Proprietary", "gpt-4.1-nano-2025-04-14",               69.10,  75.52, 62.68),
    ("Proprietary", "gpt-4-0613",                            71.55,  74.45, 68.64),
    ("Proprietary", "gpt-3.5-turbo-0125",                    57.82,  61.84, 53.81),

    # NEW entry (added by request) – first three metrics only
    ("Llama 3.1",   "meta-llama/Llama-3.1-405B-Instruct",    72.07,  72.39, 71.75),
]

# Turn rows into a DataFrame
df = pd.DataFrame(
    rows,
    columns=["License", "Model Name", "Overall Avg", "JA Avg", "EN Avg"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def assign_license(name: str, original_license: str | None) -> str | None:
    """Fill in a license when the row’s field is None."""
    name_lc = name.lower()
    if original_license:
        return original_license
    if "llama-3.1" in name_lc:
        return "Llama 3.1"
    if "llama-3.3" in name_lc:
        return "Llama 3.3"
    if "mistral" in name_lc or "qwen" in name_lc:
        return "Apache 2.0"
    if "deepseek" in name_lc or "phi" in name_lc:
        return "MIT"
    if "gpt" in name_lc:
        return "Proprietary"
    return None

df["License"] = df.apply(lambda r: assign_license(r["Model Name"], r["License"]), axis=1)

def clean_label(name: str) -> str:
    """Strip org-prefixes and shorten long identifiers."""
    if "/" in name:
        name = name.split("/", 1)[1]
    if name == "meti-geniac-405b-dpo.jury":
        name = "meti-geniac-405b-dpo"
    return name

df["Label"] = df["Model Name"].apply(clean_label)

def get_color_and_size(name: str) -> tuple[str, int]:
    n = name.lower()
    if "shisa" in n:
        return "deeppink", 60
    if "meti-geniac-405b" in n:
        return "red", 60
    if "405b-instruct" in n:
        return "skyblue", 60
    if "gpt" in n:
        return "black", 40
    return "lightgrey", 30

colors, sizes = zip(*(get_color_and_size(m) for m in df["Model Name"]))

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 8))
plt.scatter(
    df["EN Avg"], df["JA Avg"],
    c=colors, s=sizes,
    marker="o", edgecolors="white", linewidths=0.5
)

# Nudge overlapping labels
for _, row in df.iterrows():
    label, x, y, model = row["Label"], row["EN Avg"], row["JA Avg"], row["Model Name"]
    if model == "shisa-v2-qwen25-7b":
        plt.text(x + 0.3, y + 0.35, label, fontsize=6.5)
    elif model == "shisa-v2-llama3.1-8b":
        plt.text(x + 0.3, y - 0.35, label, fontsize=6.5)
    elif model in {
        "gpt-4.1-nano-2025-04-14",
        "gpt-4-turbo-2024-04-09",
        "gpt-4.1-mini-2025-04-14",
    }:
        plt.text(x + 0.3, y - 0.3, label, fontsize=6.5)
    else:
        plt.text(x + 0.3, y, label, fontsize=6.5)

plt.title("Model Performance: JA Avg vs EN Avg")
plt.xlabel("EN Avg")
plt.ylabel("JA Avg")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(f"meti-plot.png", dpi=300)

