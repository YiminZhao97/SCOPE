# Palantir Dataset Workflow

This document outlines the preprocessing and analysis workflow for the Palantir dataset.

------

## 1. Original Dataset

- **File**: `marrow_sample_scseq_counts.h5ad`
- **Description**: Original AnnData file.
  - Note: This file **does not** include cell barcodes.
  - 4142 cells × 16106 genes
  - Run5_131097901611291: DC; Run5_134936662236454: Mono; Run4_200562869397916: Ery.

------

## 2. Run Leiden and Palantir

- **Script**: `run_leiden_palantir.py`
- **Input**: `marrow_sample_scseq_counts.h5ad`
- **Output**: `data_palantir_pseudotime_leiden_clustering.h5ad`
- **Purpose**: Performs Leiden clustering and Palantir pseudotime estimation.

------

## 3. Run Mellon Denstiy Estimation

- **File**: `run_mellon.py`
- **Input**: `data_palantir_pseudotime_leiden_clustering.h5ad`
- **Output**: `data_palantir_pseudotime_leiden_clustering_mellon.h5ad`
- **Description**: Perform Mellon density estimation.

------

## Summary Flowchart

```
marrow_sample_scseq_counts.h5ad
    └── [run_leiden_palantir.py] → data_palantir_pseudotime_leiden_clustering.h5ad
            └── [run_mellon.py] → data_palantir_pseudotime_leiden_clustering_mellon.h5ad
```

------