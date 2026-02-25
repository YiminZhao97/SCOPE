"""
1. We first do leiden clustering and we aim to have 4 clusters in total: one for progenitor and 3 for terminal states
2. Run Slingshot
"""

import pandas as pd
import scanpy as sc
import anndata as ad
import os
from pyslingshot import Slingshot
from matplotlib import pyplot as plt

terminal_states = pd.Series(
    ["DC", "Mono", "Ery"],
    index=["Run5_131097901611291", "Run5_134936662236454", "Run4_200562869397916"],
)

#run leiden clustering and save the result
adata = ad.read_h5ad('/home/yzhao4/branch_point_prediction/Palantir/data/marrow_sample_scseq_counts.h5ad')
adata.var_names_make_unique() 
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40) #this step failed locally, run code on Hutch cluster
sc.tl.leiden(adata, resolution=0.15) #small resolution implies less num of clusters
adata.write_h5ad('/home/yzhao4/branch_point_prediction/Simulation/Slingshot/data_after_leiden.h5ad')

#load the result after leiden clustering
adata = ad.read_h5ad('/home/yzhao4/branch_point_prediction/Simulation/Slingshot/data_after_leiden_res15.h5ad')

adata["Run5_131097901611291"].obs['leiden'] #DC 0 
adata["Run5_134936662236454"].obs['leiden'] #Mono 2
adata["Run4_200562869397916"].obs['leiden'] #Ery 3
#progenitor 1

sc.tl.umap(adata)
sc.pl.umap(adata,color="leiden")
plt.savefig('/home/yzhao4/branch_point_prediction/Simulation/Slingshot/umap_leiden.png', dpi = 300)

#slingshot
adata.obs["celltype"] = adata.obs["leiden"]
start_node = 1
end_nodes = [0, 2, 3]
num_branches = 3
K = 4 # cluster labels

slingshot = Slingshot(adata, celltype_key="celltype", obsm_key="X_umap", start_node=start_node,
                       debug_level='verbose') #end_nodes = end_nodes,

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
custom_xlim = (-12, 12)
custom_ylim = (-12, 12)
slingshot.fit(num_epochs=1, debug_axes=axes)

pseudotime = slingshot.unified_pseudotime