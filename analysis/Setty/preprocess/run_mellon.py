import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
import palantir
import mellon
import scanpy as sc
import warnings
from numba.core.errors import NumbaDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)

ad = sc.read('/home/yzhao4/new_repo_branchpoint/Data/Palantir_bone_marrow/data/data_palantir_pseudotime_leiden_clustering.h5ad')

model = mellon.DensityEstimator()
log_density = model.fit_predict(ad.obsm["DM_EigenVectors"])

predictor = model.predict

ad.obs["mellon_log_density"] = log_density
ad.obs["mellon_log_density_clipped"] = np.clip(
    log_density, *np.quantile(log_density, [0.05, 1])
)

ad.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Palantir_bone_marrow/data/data_palantir_pseudotime_leiden_clustering_mellon.h5ad')

