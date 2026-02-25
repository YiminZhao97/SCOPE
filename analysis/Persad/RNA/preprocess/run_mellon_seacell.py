import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
import palantir
import mellon
import scanpy as sc
import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

#ad = sc.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_seacell_pipeline/cd34_multiome_rna_palantir_without_pDC_CLP.h5ad')
ad = sc.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_seacell_pipeline/cd34_multiome_rna_palantir.h5ad')

dm_res = palantir.utils.run_diffusion_maps(ad, pca_key="X_pca", n_components=20)
model = mellon.DensityEstimator()
log_density = model.fit_predict(ad.obsm["DM_EigenVectors"])

predictor = model.predict

ad.obs["mellon_log_density"] = log_density
ad.obs["mellon_log_density_clipped"] = np.clip(
    log_density, *np.quantile(log_density, [0.05, 1])
)

ad.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/cd34_multiome_atac_palantir_mellon.h5ad')
