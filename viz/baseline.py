import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# --- Configuration Variables ---
FONT_PATH = '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc'  # Confirmed from your system
FONT_SIZE_TITLE = 16
FONT_SIZE_LABELS = 12
FONT_SIZE_TICKS = 10

OUTPUT_FILENAME_EN = "shisa_baseline_en"
OUTPUT_FILENAME_JP = "shisa_baseline_jp"

# Load font directly
jp_font = fm.FontProperties(fname=FONT_PATH)

# Apply font size settings (font family handled manually)
plt.rcParams['axes.titlesize'] = FONT_SIZE_TITLE
plt.rcParams['axes.labelsize'] = FONT_SIZE_LABELS
plt.rcParams['xtick.labelsize'] = FONT_SIZE_TICKS
plt.rcParams['ytick.labelsize'] = FONT_SIZE_TICKS

# --- Data ---
data_pairs = [
    {'Shisa Model': 'shisa-v2-qwen25-7b', 'Shisa JA Avg': 71.06, 'Base Model': 'Qwen/Qwen2.5-7B-Instruct', 'Base JA Avg': 65.3},
    {'Shisa Model': 'shisa-v2-llama3.1-8b', 'Shisa JA Avg': 70.83, 'Base Model': 'meta-llama/Llama-3.1-8B-Instruct', 'Base JA Avg': 53.43},
    {'Shisa Model': 'shisa-v2-mistral-nemo-12b', 'Shisa JA Avg': 72.83, 'Base Model': 'mistralai/Mistral-Nemo-Instruct-2407', 'Base JA Avg': 58.44},
    {'Shisa Model': 'shisa-v2-unphi4-14b', 'Shisa JA Avg': 75.89, 'Base Model': 'microsoft/phi-4', 'Base JA Avg': 72.47},
    {'Shisa Model': 'shisa-v2-qwen2.5-32b', 'Shisa JA Avg': 76.97, 'Base Model': 'Qwen/Qwen2.5-32B-Instruct', 'Base JA Avg': 66.79},
    {'Shisa Model': 'shisa-v2-llama-3.3-70b', 'Shisa JA Avg': 79.72, 'Base Model': 'meta-llama/Llama-3.3-70B-Instruct', 'Base JA Avg': 72.75}
]
df = pd.DataFrame(data_pairs)

# --- Colors ---
shisa_color = (225/255, 100/255, 250/255)
base_color = (150/255, 150/255, 150/255)
line_color = base_color

# --- Plotting Function ---
def plot_shisa(df, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(12, 7))
    x_locs = np.arange(len(df))

    # Plot lines
    for i, row in df.iterrows():
        ax.plot([x_locs[i], x_locs[i]], [row['Base JA Avg'], row['Shisa JA Avg']],
                color=line_color, linestyle='-', linewidth=1)

    # Plot points
    for i, row in df.iterrows():
        ax.plot(x_locs[i], row['Base JA Avg'], marker='o', color=base_color, linestyle='None', markersize=8)
        ax.plot(x_locs[i], row['Shisa JA Avg'], marker='s', color=shisa_color, linestyle='None', markersize=8)

    ax.set_title(title, fontproperties=jp_font)
    ax.set_ylabel(ylabel, fontproperties=jp_font)
    ax.set_xticks(x_locs)
    ax.set_xticklabels(df['Shisa Model'], rotation=45, ha='right', fontproperties=jp_font)
    ax.set_xlabel("")
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)
    plt.tight_layout()

    # Save plots
    fig.savefig(f"{filename}.svg")
    fig.savefig(f"{filename}.png", dpi=300)
    plt.close(fig)

# --- Generate Plots ---
plot_shisa(df, 'Shisa V2 Improvement vs Base Models', 'JA Avg Score', OUTPUT_FILENAME_EN)
plot_shisa(df, 'ベースモデルからの改善', 'JA 平均スコア', OUTPUT_FILENAME_JP)

print("✅ Plots saved as SVG and PNG using Noto Sans CJK font.")

