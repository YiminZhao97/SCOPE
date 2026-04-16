"""
SCOPE analysis for simulated principle curve data.
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

parser = argparse.ArgumentParser(description='SCOPE analysis for simulation data')
parser.add_argument('--input_data', type=str,
                    default='/home/yzhao4/new_repo_branchpoint/Data/Simulation/Principle_curve/sim_different_skeleton/sim1/simulation_processed_data_sim1.h5ad',
                    help='Path to input h5ad file')
parser.add_argument('--output_dir', type=str,
                    default='/home/yzhao4/new_repo_branchpoint/Output/when_sim_fails/paper_res/sim1',
                    help='Output directory for results')
parser.add_argument('--omit_tail', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--label_spreading_alpha', type=float, default=0.99,
                    help='Alpha parameter for label spreading')
parser.add_argument('--iter_graph', type=int, default=100)
parser.add_argument('--initial_trees', type=int, default=100)
parser.add_argument('--trees_per_iteration', type=int, default=10)
parser.add_argument('--n_neighbors', type=int, default=10)
args = parser.parse_args()

start_time = time.time()

# Load data
data = ad.read_h5ad(args.input_data)

# Preprocessing: map column names to SCOPE class expectations
data.obs['terminal_state_cluster'] = data.obs['cluster']
data.obs['palantir_pseudotime'] = data.obs['pseudotime']

# Run SCOPE
scope = SCOPE(
    data=data,
    latent_key='X_pca',
    alpha=args.label_spreading_alpha,
    omit_tail=args.omit_tail,
    iter_graph=args.iter_graph,
    initial_trees=args.initial_trees,
    trees_per_iteration=args.trees_per_iteration,
    n_neighbors=args.n_neighbors,
    use_X=True
)

scope.prepare_data() \
     .initialize_conformal_result() \
     .build_graph() \
     .initialize_classifiers() \
     .run_all() \
     .save_results(args.output_dir)

logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds.")
