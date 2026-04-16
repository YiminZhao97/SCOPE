"""
SCOPE analysis for Retina RNA data.
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

parser = argparse.ArgumentParser(description='SCOPE analysis for Retina RNA data')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--omit_tail', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--iter_graph', type=int, default=100)
parser.add_argument('--initial_trees', type=int, default=100)
parser.add_argument('--trees_per_iteration', type=int, default=50)
args = parser.parse_args()

start_time = time.time()

# Load data
data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Retina/RNA/Retina_RNA_subset.h5ad')

# Preprocessing: filter cells and genes
data = data[~data.obs['BIP.type'].isin(['MuG', 'BIP']), data.var['highly_variable']]

# Create terminal_state_cluster from BIP.type
terminal_states = ['RGC', 'CON', 'HRZ']
data.obs['terminal_state_cluster'] = data.obs['BIP.type'].astype('category')
new_categories = list(data.obs['terminal_state_cluster'].cat.categories) + ['TBD']
data.obs['terminal_state_cluster'] = data.obs['terminal_state_cluster'].cat.set_categories(new_categories)
data.obs.loc[~data.obs['terminal_state_cluster'].isin(terminal_states), 'terminal_state_cluster'] = 'TBD'

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
