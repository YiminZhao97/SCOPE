import numpy as np
import palantir
import mellon
import scanpy as sc
import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

ad = sc.read('/home/yzhao4/new_repo_branchpoint/Data/Larrydata/Data/Complete_LARRY_dataset_adata_preprocessed_barcodes_palantir.h5ad')
ad

#dm_res = palantir.utils.run_diffusion_maps(ad, pca_key="X_pca", n_components=20)
model = mellon.DensityEstimator()
log_density = model.fit_predict(ad.obsm["DM_EigenVectors"])

predictor = model.predict

ad.obs["mellon_log_density"] = log_density
ad.obs["mellon_log_density_clipped"] = np.clip(
    log_density, *np.quantile(log_density, [0.05, 1])
)

ad.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Larrydata/Data/Complete_LARRY_dataset_adata_preprocessed_barcodes_palantir_mellon.h5ad')

