import palantir
import scanpy as sc
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.filterwarnings(action="ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings(
    action="ignore", module="scanpy", message="No data for colormapping"
)

"""
process data
"""
# Load sample data
ad = sc.read('/home/yzhao4/branch_point_prediction/Data/marrow_sample_scseq_counts.h5ad')
ad.layers['counts'] = ad.X.copy()

sc.pp.normalize_per_cell(ad)
palantir.preprocess.log_transform(ad)
sc.pp.highly_variable_genes(ad,flavor="cell_ranger") # n_top_genes=1500, 
sc.pp.pca(ad)
ad

dm_res = palantir.utils.run_diffusion_maps(ad, n_components=5)
ms_data = palantir.utils.determine_multiscale_space(ad)
sc.pp.neighbors(ad)
sc.tl.umap(ad)

"""
leiden clustering
"""
sc.tl.leiden(ad, resolution=0.9)
sc.pl.umap(ad, color=['leiden'], frameon=False, size=10, alpha=0.5)

terminal_states = pd.Series(
    ["DC", "Mono", "Ery"],
    index=["Run5_131097901611291", "Run5_134936662236454", "Run4_200562869397916"],
)

annotations = ad.obs[['leiden']].copy(deep=True).astype(str)
annotated_cluster = annotations.loc[terminal_states.index]
annotations[~annotations['leiden'].isin(annotated_cluster['leiden'].tolist())] = 'TBD'
annotations = annotations.replace({
    annotated_cluster.loc["Run5_131097901611291", 'leiden']: 'DC',
    annotated_cluster.loc["Run5_134936662236454", 'leiden']: 'Mono',
    annotated_cluster.loc["Run4_200562869397916", 'leiden']: 'Ery'
})
ad.obs['cluster'] = annotations.loc[a.obs_names]['leiden'].tolist()

sc.pl.umap(ad, color=['cluster'], frameon=False, size=10, alpha=0.5)


"""
run palantir
"""
start_cell = "Run5_164698952452459"
pr_res = palantir.core.run_palantir(
    ad, start_cell, num_waypoints=500, terminal_states=terminal_states
)

palantir.plot.plot_palantir_results(ad, s=3)
plt.show()


ad
ad.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Palantir_bone_marrow/data/data_palantir_pseudotime_leiden_clustering.h5ad')









"""
import scanpy as sc
ad = sc.read('/home/yzhao4/new_repo_branchpoint/Data/Palantir_bone_marrow/data/data_palantir_pseudotime.h5ad')
import pickle
with open('/home/yzhao4/new_repo_branchpoint/Data/Palantir_bone_marrow/data_pipeline/barcode_branchpoint_mono_ery.pkl', 'rb') as f:
    barcode = pickle.load(f)

branchpoint_data = ad[barcode]
branchpoint_data.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Palantir_bone_marrow/data_pipeline/data_branchpoint_mono_ery.h5ad')
"""
