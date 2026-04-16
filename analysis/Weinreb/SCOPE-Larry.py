"""
SCOPE analysis for LARRY dataset (Weinreb et al.)
Using the SCOPE wrapper class.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import anndata as ad
import argparse
import time
import logging

from main import SCOPE

parser = argparse.ArgumentParser(description='SCOPE analysis for LARRY data')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--omit_tail', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--iter_graph', type=int, default=100)
parser.add_argument('--initial_trees', type=int, default=100)
parser.add_argument('--trees_per_iteration', type=int, default=50)
args = parser.parse_args()

start_time = time.time()

# Load data
data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Larrydata/Data/Complete_LARRY_dataset_adata_preprocessed_barcodes_palantir_mellon.h5ad')
data = data[:, data.var['highly_variable']]

# Preprocessing: identify day 6 terminal state cells
terminal_state_celltype = ['Monocyte', 'Neutrophil', 'Baso', 'Mast', 'Erythroid', 'Meg', 'Eos', 'Ccr7_DC']
day6_terminal_mask = (data.obs['Time_Point'] == '6.0') & (data.obs['state_info'].isin(terminal_state_celltype))

# Create terminal_state_cluster column
data.obs['terminal_state_cluster'] = 'TBD'
data.obs.loc[day6_terminal_mask, 'terminal_state_cluster'] = data.obs.loc[day6_terminal_mask, 'state_info']

# Run SCOPE
scope = SCOPE(
    data=data,
    latent_key='X_pca',
    alpha=args.alpha,
    omit_tail=args.omit_tail,
    iter_graph=args.iter_graph,
    initial_trees=args.initial_trees,
    trees_per_iteration=args.trees_per_iteration,
    use_X=True
)

scope.prepare_data() \
     .initialize_conformal_result() \
     .build_graph() \
     .initialize_classifiers() \
     .run_all() \
     .save_results(args.output_dir)

logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds.")
