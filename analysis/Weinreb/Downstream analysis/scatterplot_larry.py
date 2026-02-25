import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load color palette from JSON
with open('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure2 Simulation and larry/Larry/color_palette_tab10.json', 'r') as f:
    color_palette = json.load(f)

df = pd.read_csv('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure2 Simulation and larry/Larry/scatterplot_data.csv')

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Helvetica'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Separate the data
df1 = df[df['Labels'] == 'undiff']  # Cells to be plotted in gray
df2 = df[df['Labels'] != 'undiff']  # Cells to be colored by Labels

# Filter df2 to only include cell types in the color palette
df2 = df2[df2['Labels'].isin(color_palette.keys())]

# Convert Labels in df2 to categorical for discrete coloring
df2['Labels'] = df2['Labels'].astype(str)

# Set figure size
fig, ax = plt.subplots(figsize=(8, 7))

# Plot undifferentiated cells in gray first (background layer)
ax.scatter(df1['umap1'], df1['umap2'], c='lightgray', s=5, alpha=0.5, rasterized=True)

# Plot colored cells by cell type
for cell_type in df2['Labels'].unique():
    if cell_type in color_palette:
        df_subset = df2[df2['Labels'] == cell_type]
        ax.scatter(df_subset['umap1'], df_subset['umap2'],
                  c=color_palette[cell_type], s=5, alpha=0.8,
                  label=cell_type, rasterized=True)

# Get the data range to set axis limits
all_x = df['umap1']
all_y = df['umap2']
ax.set_xlim(all_x.min(), all_x.max())
ax.set_ylim(all_y.min(), all_y.max())

# Add cell type labels offset from centroids with background
for cell_type in df2['Labels'].unique():
    if cell_type in color_palette:
        df_subset = df2[df2['Labels'] == cell_type]
        centroid_x = df_subset['umap1'].mean()
        centroid_y = df_subset['umap2'].mean()

        # Calculate the range of the cluster to determine offset
        x_range = df_subset['umap1'].max() - df_subset['umap1'].min()
        y_range = df_subset['umap2'].max() - df_subset['umap2'].min()

        # Offset the label position (adjust multiplier as needed)
        offset_x = centroid_x + x_range * 0.3
        offset_y = centroid_y + y_range * 0.3

"""
        plt.text(
            offset_x,
            offset_y,
            cell_type,
            fontsize=20,
            fontweight='bold',
            ha='center',
            va='center',
            color=color_palette[cell_type],
            #bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
        )
"""
# Add title
#plt.title('Terminal states of the Larry dataset', fontsize=20, fontweight='bold')

# Remove axes
plt.axis('off')
plt.savefig('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure2 Simulation and larry/Larry/scatterplot_larry.png',
            dpi=400, bbox_inches='tight')

plt.savefig('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure2 Simulation and larry/Larry/scatterplot_larry.pdf')

# Save the plot as PDF
plt.savefig('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure2 Simulation and larry/Larry/scatterplot_larry_text_only.pdf',
            bbox_inches='tight')
