import pandas as pd
import anndata as ad

# Load the CSV file (excluding first column)
df = pd.read_csv('/home/yzhao4/new_repo_branchpoint/Output/label_spreading/Retina/proportion_analysis/retina_filtered-peaks-genes_cone.csv', index_col=0)

# Load the AnnData objects
rna_data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Output/label_spreading/Retina/RNA/iter100_alpha0.99_omit0.1_trees10/data_complete_results.h5ad')
atac_data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Output/label_spreading/Retina/ATAC/iter100_alpha0.99_omit0.1_trees10/data_complete_results.h5ad')

# Get var_names as sets for faster lookup
rna_var_names = set(rna_data.var_names)
atac_var_names = set(atac_data.var_names)

# Check each row
df['gene_in_rna'] = df['gene'].isin(rna_var_names)
df['peak_in_atac'] = df['peak'].isin(atac_var_names)
df['both_present'] = df['gene_in_rna'] & df['peak_in_atac']

# Filter to only rows where both gene and peak are present
paired_df = df[df['both_present']].copy()

print(f"Total paired gene-peak combinations: {len(paired_df)}")

# Group peaks by gene
gene_to_peaks = paired_df.groupby('gene')['peak'].apply(list).to_dict()

print(f"Number of unique genes: {len(gene_to_peaks)}")
for gene, peaks in gene_to_peaks.items():
    print(f"  {gene}: {len(peaks)} peaks")

# Base directory for output
base_dir = '/home/yzhao4/new_repo_branchpoint/Output/label_spreading/Retina/proportion_analysis/'

# Cell type to analyze (focusing on CON as specified)
cell_type = 'CON'

# Process each gene and its corresponding peaks
for gene, peaks in gene_to_peaks.items():
    print(f"\n{'='*60}")
    print(f"Processing: {gene} - {cell_type}")
    print(f"Corresponding peaks: {len(peaks)}")
    print(f"{'='*60}")

    # Initialize data structure
    data_dict = {'iteration': []}

    # Extract values across all iterations
    for i in range(63):  # 0 to 62
        rna_key = f'feature_importance_hvg_iter_{i}'
        atac_key = f'feature_importance_iter_{i}'  # ATAC uses different key format

        # Check if keys exist in both datasets
        if rna_key in rna_data.varm and atac_key in atac_data.varm:
            data_dict['iteration'].append(i)

            # Get gene variable importance from RNA data
            if f'{gene}_{cell_type}' not in data_dict:
                data_dict[f'{gene}_{cell_type}'] = []
            gene_value = rna_data.varm[rna_key].loc[gene, cell_type]
            data_dict[f'{gene}_{cell_type}'].append(gene_value)

            # Get peak variable importance from ATAC data for each peak
            for peak in peaks:
                col_name = f'{peak}_{cell_type}'
                if col_name not in data_dict:
                    data_dict[col_name] = []
                peak_value = atac_data.varm[atac_key].loc[peak, cell_type]
                data_dict[col_name].append(peak_value)

    # Create DataFrame and save to CSV
    result_df = pd.DataFrame(data_dict)

    # Reorder columns: iteration, gene, then peaks
    cols = ['iteration', f'{gene}_{cell_type}'] + [f'{peak}_{cell_type}' for peak in peaks]
    result_df = result_df[cols]

    csv_path = base_dir + f'{gene}_{cell_type}_paired_feature_importance.csv'
    result_df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")
    print(f"Columns: iteration, {gene} importance, {len(peaks)} peak importance columns")
    print(f"Shape: {result_df.shape}")

print(f"\n{'='*60}")
print("Processing complete!")
print(f"{'='*60}")
