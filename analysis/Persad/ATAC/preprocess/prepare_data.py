import anndata as ad
import scanpy as sc
import numpy as np

"""

def filter_peaks(ATAC, n_cells = 15):
    count = np.array((ATAC.X > 0).sum(0)).squeeze()
    ATAC = ATAC[:,(count >= n_cells)]  
    
    print('After filtering, there are {} peaks'.format(ATAC.n_vars))
    ATAC.obs['library size'] = ATAC.X.sum(axis=1)
    return ATAC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adata = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data/cd34_multiome_atac.h5ad')
min_cells = int(adata.shape[0] * 0.05)
adata = filter_peaks(adata, n_cells=min_cells)

print(adata)

vae = scvi.model.PEAKVI.load("/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/VAE/ATAC/peakvi_seacell", adata=adata)
"""
def filter_peaks(ATAC, n_cells = 15):
    count = np.array((ATAC.X > 0).sum(0)).squeeze()
    ATAC = ATAC[:,(count >= n_cells)]  
    
    print('After filtering, there are {} peaks'.format(ATAC.n_vars))
    ATAC.obs['library size'] = ATAC.X.sum(axis=1)
    return ATAC

adata = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data/cd34_multiome_atac.h5ad')
min_cells = int(adata.shape[0] * 0.05)
adata = filter_peaks(adata, n_cells=min_cells)

imputed_peaks = np.load('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/VAE/ATAC/denoised_hvp.npy')
print(imputed_peaks.shape)
adata.obsm['imputed_peaks'] = imputed_peaks

latent = np.load('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/VAE/ATAC/latent_embedding.npy')
adata.obsm["X_PeakVI"] = latent

data_palantir_mellon = sc.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/cd34_multiome_rna_palantir_mellon.h5ad')
adata.obs['palantir_pseudotime'] = data_palantir_mellon.obs['palantir_pseudotime']
adata.obs['mellon_log_density'] = data_palantir_mellon.obs['mellon_log_density']

adata.obsm['X_umap'] = data_palantir_mellon.obsm['X_umap']

adata_subset = adata[~adata.obs['celltype'].isin(['CLP', 'cDC'])]
print(adata_subset.shape)
adata_subset.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/cd34_multiome_atac_palantir_mellon_without_cDC_CLP.h5ad')