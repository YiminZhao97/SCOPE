import scanpy as sc
import scanpy.external as sce
import anndata as ad
import os
import matplotlib.pyplot as plt
import seaborn as sns
import palantir
from scipy.spatial import distance_matrix
import numpy as np
import pandas as pd

"""
We are following the procedure in https://github.com/dpeerlab/SEACells/blob/main/notebooks/SEACell_tf_activity.ipynb
"""

#read in data
os.chdir('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC')
adata = sc.read('./data/cd34_multiome_rna.h5ad') #still count matrix

#sc.pl.scatter(adata, basis='umap', color='celltype', frameon=False)

# Saving count data
adata.layers["counts"] = adata.X.copy()
# Normalizing to median total counts
sc.pp.normalize_total(adata)
# Logarithmize the data
sc.pp.log1p(adata)
#do pca
sc.tl.pca(adata, n_comps=30, use_highly_variable=True, svd_solver='arpack')

dm_res = palantir.utils.run_diffusion_maps(adata, n_components=10)
ms_data = palantir.utils.determine_multiscale_space(adata)

terminal_states = pd.Series(['cDC', 'CLP', 'pDC', 'Mega', 'Ery', 'Mono'], 
                           index=['cd34_multiome_rep1#GCACATTAGTTGTCAA-1', 'cd34_multiome_rep2#AAACGTACACCTGCCT-1', 
                                  'cd34_multiome_rep2#ATTTAGCCAATATACC-1', 'cd34_multiome_rep2#GTACGTAGTTGTAAAC-1', 
                                  'cd34_multiome_rep2#CTTCGCGTCAGCATTA-1', 'cd34_multiome_rep2#TTTCCTGAGCTGTAAC-1'])

start_cell = 'cd34_multiome_rep2#TGTGGCGGTAGGATTT-1'

pr_res = palantir.core.run_palantir(adata, start_cell,
                                    terminal_states=terminal_states.index)

print(adata)
adata.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_seacell_pipeline/cd34_multiome_rna_palantir.h5ad')