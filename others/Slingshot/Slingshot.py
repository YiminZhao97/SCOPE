"""
Palantir tutorial only gives us annotations for 3 cells(DC, Mono, Ery)
We run leiden clustering with high resolution on the whole dataset
Clusters with those 3 cells will be annotated
12 clusters in total,  DC:cluster 9 Mono:cluster 5 Ery:cluster 10
"""
import pandas as pd
import scanpy as sc
import anndata as ad
import os
from pyslingshot import Slingshot
from matplotlib import pyplot as plt

adata = ad.read_h5ad('/home/yzhao4/branch_point_prediction/Palantir/data/palantir_tutorial_data_after_leiden.h5ad')
sc.pp.neighbors(adata)
sc.tl.umap(adata)

adata.obs["celltype"] = adata.obs["leiden"]

#double check umap
sc.pl.umap(
    adata,
    color="celltype"
)
#plt.savefig('/home/yzhao4/branch_point_prediction/Simulation/palantir_umap.png', dpi = 300)

start_node = 0
end_nodes = [9, 5, 10]
num_branches = 3
K = 12 # cluster labels

slingshot = Slingshot(adata, celltype_key="celltype", obsm_key="X_umap", start_node=start_node,
                      end_nodes = end_nodes, debug_level='verbose')

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
custom_xlim = (-12, 12)
custom_ylim = (-12, 12)
slingshot.fit(num_epochs=1, debug_axes=axes)

pseudotime = slingshot.unified_pseudotime

