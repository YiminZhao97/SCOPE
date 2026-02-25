import anndata as ad
import scanpy as sc
import scvi
import torch
import numpy as np
import os

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
#sc.pp.filter_genes(adata, min_cells=min_cells)

print(adata)

scvi.model.PEAKVI.setup_anndata(adata)
vae = scvi.model.PEAKVI(adata)
vae.train()

model_dir = os.path.join('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/VAE', "peakvi_seacell")
vae.save(model_dir, overwrite=True)

imputed_peaks = vae.get_accessibility_estimates()
print(imputed_peaks.shape)

adata.obsm['imputed_peaks'] = imputed_peaks

#np.save('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/VAE/denoised_hvp.npy', imputed_peaks)
#latent = vae.get_latent_representation()
#print(latent.shape)
#np.save('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/VAE/latent_embedding.npy', latent)

#load in palantir and mellon results
data_palantir_mellon = sc.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/cd34_multiome_rna_palantir_mellon.h5ad')

adata.obs['palantir_pseudotime'] = data_palantir_mellon.obs['palantir_pseudotime']
adata.obs['mellon_log_density'] = data_palantir_mellon.obs['mellon_log_density']

adata.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/cd34_multiome_atac_palantir_mellon_peakvi.h5ad')
