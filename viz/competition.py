import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

# === Configuration Variables ===
FONT_PATH = '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc'  # Adjust if needed
font_prop = fm.FontProperties(fname=FONT_PATH)  # Or set to None if not using

OUTPUT_FILENAME_EN = "shisa_competition_en"
OUTPUT_FILENAME_JP = "shisa_competition_jp"

# Custom colors
shisa_ja_color = (255/255, 99/255, 99/255)
shisa_en_color = (99/255, 190/255, 255/255)
base_ja_color = (255/255, 220/255, 240/255)
base_en_color = (172/255, 215/255, 240/255)

CLASS_COLOR = (200/255, 200/255, 200/255)

# === Hardcoded model data ===
model_data = [
    {'Short Model Name': 'Mistral Nemo 2407', 'Class': '13B', 'JA Avg': 58.44, 'EN Avg': 48.07},
    {'Short Model Name': 'Qwen 2.5 14B', 'Class': '13B', 'JA Avg': 71.02, 'EN Avg': 62.54},
    {'Short Model Name': 'CA Mistral Nemo JA 2408', 'Class': '13B', 'JA Avg': 71.12, 'EN Avg': 48.0},
    {'Short Model Name': 'Phi 4', 'Class': '13B', 'JA Avg': 72.47, 'EN Avg': 61.14},
    {'Short Model Name': 'Shisa V2 Mistral Nemo 12B', 'Class': '13B', 'JA Avg': 72.83, 'EN Avg': 53.33},
    {'Short Model Name': 'Gemma 3 12B', 'Class': '13B', 'JA Avg': 75.15, 'EN Avg': 62.1},
    {'Short Model Name': 'Shisa V2 Unphi 4 14B', 'Class': '13B', 'JA Avg': 75.89, 'EN Avg': 60.1},
    {'Short Model Name': 'Mistral Small 3.1', 'Class': '30B', 'JA Avg': 72.03, 'EN Avg': 65.15},
    {'Short Model Name': 'Qwen 2.5 32B', 'Class': '30B', 'JA Avg': 73.35, 'EN Avg': 66.67},
    {'Short Model Name': 'ABEJA Qwen 2.5 32B JA v0.1', 'Class': '30B', 'JA Avg': 74.14, 'EN Avg': 65.7},
    {'Short Model Name': 'Shisa V2 Qwen 2.5 32B', 'Class': '30B', 'JA Avg': 76.97, 'EN Avg': 67.41},
    {'Short Model Name': 'Gemma 3 27b', 'Class': '30B', 'JA Avg': 80.05, 'EN Avg': 65.95},
    {'Short Model Name': 'Llama 3.3 70B', 'Class': '70B', 'JA Avg': 72.75, 'EN Avg': 71.48},
    {'Short Model Name': 'CA Llama 3.1 70B JA', 'Class': '70B', 'JA Avg': 73.67, 'EN Avg': 64.47},
    {'Short Model Name': 'Tulu 3 70B', 'Class': '70B', 'JA Avg': 74.64, 'EN Avg': 64.48},
    {'Short Model Name': 'Swallow 70B v0.4', 'Class': '70B', 'JA Avg': 75.59, 'EN Avg': 61.03},
    {'Short Model Name': 'Qwen 2.5 72B', 'Class': '70B', 'JA Avg': 77.57, 'EN Avg': 68.12},
    {'Short Model Name': 'Shisa V2 Llama 3.3 70b', 'Class': '70B', 'JA Avg': 79.72, 'EN Avg': 67.71},
    {'Short Model Name': 'Llama 3.1 8B', 'Class': '8B', 'JA Avg': 53.43, 'EN Avg': 53.88},
    {'Short Model Name': 'llm-jp 3 7.2B', 'Class': '8B', 'JA Avg': 56.05, 'EN Avg': 23.46},
    {'Short Model Name': 'Llama 3 ELYZA JP 8B', 'Class': '8B', 'JA Avg': 60.92, 'EN Avg': 39.09},
    {'Short Model Name': 'Qwen 2.5 7B', 'Class': '8B', 'JA Avg': 65.3, 'EN Avg': 58.11},
    {'Short Model Name': 'Swallow 8B v0.3', 'Class': '8B', 'JA Avg': 67.44, 'EN Avg': 42.2},
    {'Short Model Name': 'Shisa V2 Llama 3.1 8B', 'Class': '8B', 'JA Avg': 70.83, 'EN Avg': 54.75},
    {'Short Model Name': 'Shisa V2 Qwen 2.5 7B', 'Class': '8B', 'JA Avg': 71.06, 'EN Avg': 54.86}
]

# === Main Plotting Function ===
def render_model_comparison_chart_with_custom_labels(
    model_data,
    output_basename="shisa_improvement_plot",
    font_prop=None,
    title="Model Comparison",
    ylabel="Average Score",
    label_ja="JA Avg",
    label_en="EN Avg",
    class_labels=("7–8B", "12–14B", "24–32B", "70–72B"),
    class_order=("8B", "13B", "30B", "70B"),
    label_fontsize=10,
    label_fontcolor="black",
    score_label_fontsize=8,
    score_label_fontcolor="gray",
    class_label_y=84,
    ja_legend=0
):
    if ja_legend:
        LEGEND_TITLE = '言語'
        LEGEND_JA = "日本語"
        LEGEND_EN = "英語"
    else:
        LEGEND_TITLE = 'Language'
        LEGEND_JA = label_ja
        LEGEND_EN = label_en

    model_data.sort(key=lambda x: (class_order.index(x["Class"]), x[label_ja]))

    bar_width = 0.4
    spacing_within = 1
    spacing_between_classes = 2.5

    plt.figure(figsize=(18, 10))
    ax = plt.gca()

    x_pos, bar_values, bar_colors = [], [], []
    xtick_labels, xtick_positions = [], []
    current_x = 0
    class_label_positions = []

    for i, class_label in enumerate(class_order):
        class_models = [m for m in model_data if m["Class"] == class_label]
        class_start_x = current_x

        for model in class_models:
            en_x = current_x
            ja_x = current_x + bar_width
            is_shisa = "shisa" in model["Short Model Name"].lower()

            en_color = shisa_en_color if is_shisa else base_en_color
            ja_color = shisa_ja_color if is_shisa else base_ja_color

            x_pos.extend([en_x, ja_x])
            bar_values.extend([model[label_en], model[label_ja]])
            bar_colors.extend([en_color, ja_color])
            xtick_positions.append(current_x + bar_width * 0.5)
            xtick_labels.append(model["Short Model Name"])

            ax.text(
                ja_x + bar_width * 0.05,
                model[label_ja] + 1.2,
                f"{model[label_ja]:.1f}",
                ha="center",
                va="bottom",
                fontsize=score_label_fontsize,
                color=score_label_fontcolor,
                fontproperties=font_prop
            )

            current_x += spacing_within

        class_label_positions.append(class_start_x)
        current_x += spacing_between_classes

    ax.bar(x_pos, bar_values, width=bar_width, color=bar_colors)
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontproperties=font_prop)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel, fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop)

    legend_handles = [
        Patch(color=shisa_ja_color, label=LEGEND_JA),
        Patch(color=shisa_en_color, label=LEGEND_EN),
    ]
    legend = ax.legend(handles=legend_handles, title=LEGEND_TITLE, prop=font_prop)
    legend.get_title().set_fontproperties(font_prop)


    for x, class_label in zip(class_label_positions, class_labels):
        ax.text(
            x,
            class_label_y,
            class_label,
            ha="left",
            va="bottom",
            fontsize=label_fontsize,
            color=label_fontcolor,
            fontproperties=font_prop
        )

    plt.tight_layout()
    plt.savefig(f"{output_basename}.png", dpi=300)
    plt.savefig(f"{output_basename}.svg", format="svg")
    plt.close()

# === Example call ===
render_model_comparison_chart_with_custom_labels(
    model_data=model_data,
    output_basename=OUTPUT_FILENAME_EN,
    font_prop=font_prop,
    title="Model Comparison by Class",
    ylabel="Average Score",
    label_ja="JA Avg",
    label_en="EN Avg",
    class_labels=("8B", "13B", "30B", "70B"),
    label_fontsize=24,
    label_fontcolor=CLASS_COLOR,
    score_label_fontsize=0,
    score_label_fontcolor="dimgray",
    class_label_y=84
)

render_model_comparison_chart_with_custom_labels(
    model_data=model_data,
    output_basename=OUTPUT_FILENAME_JP,
    font_prop=font_prop,
    title="モデル比較",
    ylabel="平均スコア",
    label_ja="JA Avg",
    label_en="EN Avg",
    class_labels=("8B", "13B", "30B", "70B"),
    label_fontsize=24,
    label_fontcolor=CLASS_COLOR,
    score_label_fontsize=0,
    score_label_fontcolor="dimgray",
    class_label_y=84,
    ja_legend = 1
)
