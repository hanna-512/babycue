import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
})

# ---------------------------------------------------------
# Figure 1: MD5 Waveform Deduplication Audit (1,126 -> 638)
# ---------------------------------------------------------
classes = ["Hunger", "Pain_Discomfort", "Sleepiness"]
raw_counts = [397, 587, 142]
clean_counts = [397, 226, 15]
dup_counts = [r - c for r, c in zip(raw_counts, clean_counts)]

x = np.arange(len(classes))
width = 0.36

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
b1 = ax.bar(x - width / 2, raw_counts, width, label="Raw Files Scanned (Total = 1,126)", color="#94a3b8")
b2 = ax.bar(x + width / 2, clean_counts, width, label="Unique Clips Kept After MD5 Deduplication (Total = 638)", color="#2563eb")

ax.set_ylabel("Number of Audio Recordings")
ax.set_title("Dataset Integrity Audit: Raw vs. MD5-Deduplicated Recordings")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 660)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(frameon=True)

for bar in b1:
    h = bar.get_height()
    ax.annotate(f"{h}", (bar.get_x() + bar.get_width() / 2, h + 10), ha="center", fontweight="bold", color="#475569")
for bar, dup in zip(b2, dup_counts):
    h = bar.get_height()
    label = f"{h}\n(-{dup} dups)" if dup > 0 else f"{h}\n(0 dups)"
    ax.annotate(label, (bar.get_x() + bar.get_width() / 2, h + 10), ha="center", fontweight="bold", color="#1d4ed8", fontsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig1_dataset_deduplication.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 2: Model Progression & Baseline Benchmark
# ---------------------------------------------------------
models = [
    "Dummy Majority\nBaseline ('Hunger')",
    "FYP1 Legacy\nManual MFCCs",
    "FYP2 YAMNet + PCA(64)\n(638 Multi-Format)",
    "FYP2 YAMNet + PCA(64)\n(586 High-SNR WAV)"
]
overall_acc = [62.23, 33.00, 67.55, 69.45]
bal_acc = [33.33, 33.33, 49.87, 56.41]
macro_f1_pct = [25.57, 28.00, 48.88, 57.38]

x = np.arange(len(models))
w = 0.25

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
r1 = ax.bar(x - w, overall_acc, w, label="Overall Accuracy (%)", color="#1e3a8a")
r2 = ax.bar(x, bal_acc, w, label="Balanced Accuracy (%)", color="#0d9488")
r3 = ax.bar(x + w, macro_f1_pct, w, label="Macro F1-Score (×100)", color="#f59e0b")

ax.set_ylabel("Score (%)")
ax.set_title("CrySense Model Benchmark (5-Fold Stratified CV, Strict Fold Isolation)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 82)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(frameon=True, loc="upper left")

for rects in (r1, r2, r3):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", (rect.get_x() + rect.get_width() / 2, h + 1.2), ha="center", fontsize=8.5, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig2_benchmark_comparison.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 3: 3x3 Confusion Matrix Heatmap (Loaded from CSV)
# ---------------------------------------------------------
cm_df = pd.read_csv(os.path.join(RESULTS_DIR, "yamnet_confusion_matrix.csv"), index_col=0)
cm = cm_df.to_numpy()
row_sums = cm.sum(axis=1, keepdims=True)
cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

annot = np.empty(cm.shape, dtype=object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=300)
sns.heatmap(
    cm_pct,
    annot=annot,
    fmt="",
    cmap="Blues",
    cbar_kws={"label": "Row-Normalized Recall (%)"},
    xticklabels=["Hunger", "Pain_Discomfort", "Sleepiness"],
    yticklabels=["Hunger", "Pain_Discomfort", "Sleepiness"],
    vmin=0,
    vmax=85,
    ax=ax,
    annot_kws={"size": 11, "weight": "bold"},
)
ax.set_xlabel("Predicted Class", fontweight="bold")
ax.set_ylabel("Actual Class", fontweight="bold")
ax.set_title(f"Unseen 5-Fold Confusion Matrix (N = {int(cm.sum())} Clips)")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig3_confusion_matrix.png"), dpi=300)
plt.close()

# ---------------------------------------------------------
# Figure 4: Per-Class Precision, Recall, and F1-Score
# ---------------------------------------------------------
# Computed directly from yamnet_confusion_matrix.csv
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)

x = np.arange(len(classes))
w = 0.25

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
p_bar = ax.bar(x - w, precision, w, label="Precision", color="#3b82f6")
r_bar = ax.bar(x, recall, w, label="Recall", color="#10b981")
f_bar = ax.bar(x + w, f1, w, label="F1-Score", color="#6366f1")

ax.set_ylabel("Score (0.00 – 1.00)")
ax.set_title(f"Per-Class Performance Breakdown (N = {int(cm.sum())} Clips)")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(n={int(s)})" for c, s in zip(classes, cm.sum(axis=1))])
ax.set_ylim(0, 0.95)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(frameon=True)

for rects in (p_bar, r_bar, f_bar):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h + 0.02), ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig4_per_class_metrics.png"), dpi=300)
plt.close()

print("Saved all 4 thesis figures to 'results/':")
for fname in sorted(os.listdir(RESULTS_DIR)):
    if fname.endswith(".png"):
        print(" ->", os.path.join(RESULTS_DIR, fname))