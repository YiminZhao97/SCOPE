# SCOPE: Localizing fate-decision states and their regulatory drivers in single-cell differentiation

## SCOPE

SCOPE is a framework for localizing fate-decision states and their regulatory drivers in single-cell data. SCOPE models differentiation as a progression where cells transition from multipotent to committed states, and formalizes this process using conformal prediction to quantify fate uncertainty. By generating calibrated prediction sets of plausible terminal fates for each cell, SCOPE captures both the continuity of cellular trajectories and the discrete boundaries of lineage commitment. SCOPE is designed to work with high-dimensional single-cell and multi-omic data, enabling the identification of branchpoints and the timing of regulatory events such as epigenetic priming.

## Installation

```bash
# Add SCOPE to your Python path
export PYTHONPATH="/path/to/SCOPE:$PYTHONPATH"

# Or add in your script
import sys
sys.path.append('/path/to/SCOPE')
```

## Quick Start

```python
import anndata as ad
from main import SCOPE

# Load your data (must have Palantir pseudotime already computed)
data = ad.read_h5ad('your_data.h5ad')

# Create terminal_state_cluster column
# Set known terminal states to their labels, unknown cells to 'TBD'
data.obs['terminal_state_cluster'] = ...

# Initialize SCOPE
scope = SCOPE(
    data=data,
    feature_key='imputed_hvg',      # Key in data.obsm with features
    latent_key='vae_latent_space',  # Key in data.obsm with latent space
    alpha=0.1,                       # Label spreading parameter
    iter_graph=100,                  # Label spreading iterations
    initial_trees=100                # Initial RF trees
)

# Run SCOPE pipeline
scope.prepare_data() \
     .initialize_conformal_result() \
     .build_graph() \
     .initialize_classifiers() \
     .run_all() \
     .save_results('output_dir')
```

## Required Data Format

Your AnnData object must contain:

### `.obs` (cell metadata)
- `terminal_state_cluster`: Cell fate labels
  - Known terminal states: 'Monocyte', 'Neutrophil', etc.
  - Unknown cells: 'TBD' (To Be Determined)
- `palantir_pseudotime`: Pseudotime values from Palantir

### `.obsm` (multidimensional annotations)
- Feature matrix (e.g., `imputed_hvg`): Cell x genes matrix for classification
  - **OR** use `.X` directly if no imputation available (set `use_X=True`)
- Latent space (e.g., `vae_latent_space`): Low-dimensional embedding for graph construction

See [API.md](API.md) for detailed class methods, parameters, output format, and example datasets.

## Citation

SCOPE manuscript is available from [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.04.07.717037v1.supplementary-material). If you use SCOPE for your work, please cite our paper.

```
@article{zhao2026scope,
  title={SCOPE: Localizing fate-decision states and their regulatory drivers in single-cell differentiation},
  author={Zhao, Yimin and Finkbeiner, Connor and Setty, Manu and Lin, Kevin},
  journal={bioRxiv},
  pages={2026--04},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact [ymzhao97@uw.edu](mailto:ymzhao97@uw.edu)
