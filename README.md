# SCOPE: Single-Cell Ontogeny Prediction Engine

SCOPE is a Python framework for predicting cell fate trajectories using conformal prediction and iterative semi-supervised learning.

## Overview

SCOPE combines:
- **Random Forest classifiers** for cell fate prediction
- **Label spreading** on k-nearest neighbor graphs
- **Conformal prediction** for uncertainty quantification
- **Iterative learning** with progressive recruitment of unlabeled cells

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
- Feature matrix (e.g., `imputed_hvg`): Cell × genes matrix for classification
  - **OR** use `.X` directly if no imputation available (set `use_X=True`)
- Latent space (e.g., `vae_latent_space`): Low-dimensional embedding for graph construction

## SCOPE Class Methods

### Initialization

```python
scope = SCOPE(
    data,                      # AnnData object
    feature_key='imputed_hvg', # Feature matrix key (ignored if use_X=True)
    latent_key='vae_latent_space',  # Latent space key
    alpha=0.1,                 # Label spreading alpha
    omit_tail=0,               # Prediction set tail omission
    iter_graph=100,            # Label spreading iterations
    initial_trees=100,         # Initial RF trees
    trees_per_iteration=50,    # Trees added per iteration
    n_neighbors=10,            # k-NN graph neighbors
    use_X=False                # If True, use data.X instead of data.obsm[feature_key]
)
```

### Core Methods

#### `prepare_data()`
Prepares data for SCOPE analysis:
- Identifies TBD (unlabeled) cells
- Creates dummy labels for terminal states
- Sorts cells by pseudotime
- Initializes visit tracking

**Returns:** `self` (for method chaining)

#### `initialize_conformal_result()`
Initializes conformal prediction structures:
- Sets recruitment size (√n_cells)
- Creates result dictionaries for qhat, prediction sets, sizes

**Returns:** `self` (for method chaining)

#### `build_graph()`
Builds k-nearest neighbor graph for label spreading:
- Constructs graph from latent space
- Uses specified number of neighbors
- Euclidean distance metric

**Returns:** `self` (for method chaining)

#### `initialize_classifiers()`
Creates random forest classifiers:
- One classifier per terminal state
- Initialized with specified number of trees
- Binary classification setup

**Returns:** `self` (for method chaining)

#### `run_scope()`
Runs **one iteration** of SCOPE prediction:
1. Updates classifiers (adds trees if not first iteration)
2. Selects test cells based on pseudotime
3. Trains random forest classifiers
4. Applies label spreading on graph
5. Computes conformal prediction sets
6. Updates cell labels

**Returns:** `bool` - True if successful, False if no cells left

**Use case:** When you want control over each iteration
```python
scope.prepare_data() \
     .initialize_conformal_result() \
     .build_graph() \
     .initialize_classifiers()

# Run iterations manually
while scope.run_scope():
    print(f"Completed iteration {scope.iteration}")
    # Do custom analysis between iterations
```

#### `run_all()`
Runs SCOPE until all cells are processed:
- Repeatedly calls `run_scope()` until complete
- Tracks total execution time
- Prints summary

**Returns:** `self` (for method chaining)

#### `save_results(output_dir)`
Saves SCOPE results to disk:
- `data_complete_results.h5ad`: Annotated data with predictions
- `conformal_result.pkl`: Conformal prediction results

**Parameters:**
- `output_dir` (str): Directory to save results

## Output

### Updated AnnData object

After running SCOPE, your data object contains:

#### `.obs` columns
- `visit`: 1 if cell has been labeled, 0 if still TBD
- `iteration_recruited`: Which iteration cell was labeled (-1 for initial)
- `terminal_state_cluster`: Updated with predictions

#### `.obsm` matrices
- `dummy_label`: Probabilistic labels (cells × terminal states)

#### Variable importance
- Stored per iteration in `.varm` or `.uns`

### Conformal results dictionary

```python
conformal_result = {
    'recruitment_size': int,           # Cells recruited per iteration
    'qhat': [float, ...],              # Threshold per iteration
    'size': DataFrame,                 # Prediction set sizes
    'prediction_set': DataFrame,       # Prediction sets per cell
    'prob_test': DataFrame            # Test probabilities
}
```

## Parameters Guide

### Label Spreading
- **`alpha`** (default: 0.1): Controls label smoothness
  - Lower values: More spreading, smoother labels
  - Higher values: Less spreading, preserve initial predictions
  - Range: [0, 1]

- **`iter_graph`** (default: 100): Label spreading iterations
  - More iterations: Better convergence
  - Typical range: 50-200

### Random Forest
- **`initial_trees`** (default: 100): Starting number of trees
  - More trees: Better initial predictions, slower
  - Typical range: 50-200

- **`trees_per_iteration`** (default: 50): Trees added per iteration
  - Gradual increase in model complexity
  - Typical range: 20-100

### Graph Construction
- **`n_neighbors`** (default: 10): k-NN graph neighbors
  - More neighbors: Smoother predictions, higher computation
  - Typical range: 5-30

### Conformal Prediction
- **`omit_tail`** (default: 0): Probability mass to omit from prediction sets
  - Removes unlikely predictions
  - Range: [0, 0.5]

## Example Datasets

### Palantir Bone Marrow (Setty et al.)
```python
# See: analysis/Setty/SCOPE-Setty-refactored.py
data = ad.read_h5ad('palantir_bone_marrow.h5ad')
data.obs['terminal_state_cluster'] = data.obs['cluster']

scope = SCOPE(
    data=data,
    feature_key='imputed_hvg',
    latent_key='vae_latent_space'
)
```

### HSPC Multiome (Persad et al.)
```python
# See: analysis/Persad/RNA/SCOPE-Persad-RNA-refactored.py
data = ad.read_h5ad('cd34_multiome_rna.h5ad')
# Set terminal states: DC, Ery, Mono
# Set others to TBD

scope = SCOPE(
    data=data,
    feature_key='imputed_hvg',
    latent_key='vae_latent_space'
)
```

### LARRY (Weinreb et al.)
```python
# See: analysis/Weinreb/SCOPE-Larry-refactored.py
data = ad.read_h5ad('larry_dataset.h5ad')
# Use day 6 terminal states as labeled
# Rest as TBD

# Option 1: Use data.X directly (no imputation)
scope = SCOPE(
    data=data,
    latent_key='X_pca',
    use_X=True  # Use data.X instead of data.obsm
)

# Option 2: If you have feature matrix in obsm
scope = SCOPE(
    data=data,
    feature_key='highly_variable_genes',
    latent_key='X_pca',
    use_X=False
)
```

## Workflow

```
1. Prepare Data
   ↓
   - Load AnnData with Palantir results
   - Create terminal_state_cluster column
   - Mark known terminal states
   - Mark unknown cells as 'TBD'

2. Initialize SCOPE
   ↓
   - Set feature and latent space keys
   - Configure parameters (alpha, trees, etc.)

3. Prepare Data
   ↓
   scope.prepare_data()
   - Sort by pseudotime
   - Create dummy labels
   - Initialize tracking

4. Initialize Conformal Result
   ↓
   scope.initialize_conformal_result()
   - Set recruitment size
   - Create result structures

5. Build Graph
   ↓
   scope.build_graph()
   - Construct k-NN graph from latent space

6. Initialize Classifiers
   ↓
   scope.initialize_classifiers()
   - Create RF classifiers for each terminal state

7. Run SCOPE
   ↓
   scope.run_all()  # or scope.run_scope() for single iteration

   For each iteration:
   - Update classifiers (add trees)
   - Select test cells by pseudotime
   - Train RF on labeled cells
   - Apply label spreading
   - Compute conformal prediction sets
   - Update labels
   - Recruit new cells

8. Save Results
   ↓
   scope.save_results('output_dir')
   - Save annotated data
   - Save conformal results
```

## Tips

1. **Start with defaults**: The default parameters work well for most datasets

2. **Pseudotime matters**: Ensure Palantir pseudotime is well-calibrated

3. **Terminal states**: Be conservative - only mark cells you're confident about

4. **Monitor progress**: Use `run_scope()` in a loop to track each iteration

5. **Feature selection**: Use highly variable genes or imputed features

6. **Latent space**: VAE or PCA embeddings work well for graph construction

## Citation

If you use SCOPE in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please [contact information or link to issues page]
