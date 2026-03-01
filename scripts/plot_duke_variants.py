"""
Generate a grouped bar chart comparing Rank@1 and mAP across Duke variants.
Output: report/pics/duke_variants_bar.pdf
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = 9

# --- data ---
labels = [
    "Market→Duke\n(zero-shot)",
    "ResNet-50\n(baseline)",
    "DenseNet",
    "HRNet",
    "Circle loss",
    "Instance loss",
    "Triplet loss",
]

rank1 = [0.3299, 0.7935, 0.8151, 0.8389, 0.7828, 0.8007, 0.7931]
mAP   = [0.1700, 0.6174, 0.6484, 0.6946, 0.6128, 0.6232, 0.6232]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7.5, 3.8))

bars_r1 = ax.bar(x - width / 2, rank1, width, label="Rank@1", color="#4c72b0", zorder=3)
bars_map = ax.bar(x + width / 2, mAP,   width, label="mAP",    color="#dd8452", zorder=3)

# value annotations
for bar in bars_r1:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.008,
        f"{bar.get_height():.4f}",
        ha="center", va="bottom", fontsize=7, color="#4c72b0",
    )
for bar in bars_map:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.008,
        f"{bar.get_height():.4f}",
        ha="center", va="bottom", fontsize=7, color="#dd8452",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("DukeMTMC-reID: Rank@1 and mAP Across Variants")
ax.legend(loc="upper left")
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

fig.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "..", "report", "pics")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "duke_variants_bar.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out_path)}")
