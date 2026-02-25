import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
data_folder = "/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure3 Larry/res variable importance"
mono_path = os.path.join(data_folder, "larry_variable_importance_monocyte.csv")
neu_path = os.path.join(data_folder, "larry_variable_importance_neutrophil.csv")

MONO_COL = "weighted_importance_monocyte"
NEU_COL = "weighted_importance_neutrophil"

highlight_genes = ["Dab2", "Gfi1", "Gata2"]
highlight_colors = {"Dab2": "#43729C", "Gfi1": "#DC143C", "Gata2": "#55AFA9"} #, "Gata2": "#E933F6, "Gfi1": "#D38640" 

# ---------------- HELPER ----------------
def prep_curve(path, value_col, lineage_name):
    df = pd.read_csv(path, index_col=0)
    vals = pd.to_numeric(df[value_col], errors="coerce").values
    finite = np.isfinite(vals)
    if finite.any():
        max_f = np.nanmax(vals[finite])
        vals[~finite] = 2.0 * max_f
    s = pd.Series(vals, index=df.index, name="value").sort_values(ascending=False)
    out = s.reset_index().rename(columns={"index": "gene"})
    out["rank"] = np.arange(1, len(out) + 1)
    out["lineage"] = lineage_name
    return out

# ---------------- DATA ----------------
mono = prep_curve(mono_path, MONO_COL, "Monocyte")
neu = prep_curve(neu_path, NEU_COL, "Neutrophil")

# Filter to first 500 samples
mono = mono.head(500)
neu = neu.head(500)

# ---------------- PLOT ----------------
# Set font parameters
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))

# Highlight specific genes
def annotate_highlights(ax, df, color_base):
    sub = df[df["gene"].isin(highlight_genes)]
    for _, row in sub.iterrows():
        gene = row["gene"]
        color = highlight_colors.get(gene, color_base)
        ax.scatter(row["rank"], row["value"], s=100, c=color,
                  edgecolors="black", linewidths=2, zorder=5, alpha=0.9)
        ax.annotate(gene, (row["rank"], row["value"]),
                     xytext=(5, 5), textcoords="offset points",
                     fontsize=20, color=color,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color, alpha=0.8, linewidth=2))

# Plot 1: Monocyte
ax1.scatter(mono["rank"], mono["value"], s=8, alpha=0.7, color="#1f77b4")
annotate_highlights(ax1, mono, "#1f77b4")
ax1.set_title("Monocyte", fontsize=20)
ax1.set_xlabel("Rank (decreasing value)", fontsize=20)
ax1.set_ylabel("teststat", fontsize=20)
ax1.tick_params(labelsize=20)

# Plot 2: Neutrophil
ax2.scatter(neu["rank"], neu["value"], s=8, alpha=0.7, color="#ff7f0e")
annotate_highlights(ax2, neu, "#ff7f0e")
ax2.set_title("Neutrophil", fontsize=20)
ax2.set_xlabel("Rank (decreasing value)", fontsize=20)
ax2.set_ylabel("teststat", fontsize=20)
ax2.tick_params(labelsize=20)

plt.tight_layout()
output_path = os.path.join(data_folder, "gene_plot_mono_neu.pdf")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
