"""
SCOPE analysis for Persad HSPC multiome RNA data.
Using the SCOPE wrapper class.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import anndata as ad
import argparse
import time
import logging

from main import SCOPE

parser = argparse.ArgumentParser(description='SCOPE analysis for Persad RNA data')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--omit_tail', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--iter_graph', type=int, default=100)
parser.add_argument('--initial_trees', type=int, default=100)
parser.add_argument('--trees_per_iteration', type=int, default=50)
args = parser.parse_args()

start_time = time.time()

# Load data
data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_seacell_pipeline/cd34_multiome_rna_palantir_without_cDC_CLP_vae_withallTFs.h5ad')

# Preprocessing: rename pDC to DC and create terminal_state_cluster
data.obs['celltype'] = data.obs['celltype'].replace({'pDC': 'DC'})
terminal_states = ['DC', 'Ery', 'Mono']
data.obs['terminal_state_cluster'] = data.obs['celltype'].astype('category')
new_categories = list(data.obs['terminal_state_cluster'].cat.categories) + ['TBD']
data.obs['terminal_state_cluster'] = data.obs['terminal_state_cluster'].cat.set_categories(new_categories)
data.obs.loc[~data.obs['terminal_state_cluster'].isin(terminal_states), 'terminal_state_cluster'] = 'TBD'

# Run SCOPE
scope = SCOPE(
    data=data,
    feature_key='imputed_hvg',
    latent_key='vae_latent_space',
    alpha=args.alpha,
    omit_tail=args.omit_tail,
    iter_graph=args.iter_graph,
    initial_trees=args.initial_trees,
    trees_per_iteration=args.trees_per_iteration
)

scope.prepare_data() \
     .initialize_conformal_result() \
     .build_graph() \
     .initialize_classifiers() \
     .run_all() \
     .save_results(args.output_dir)

logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds.")
