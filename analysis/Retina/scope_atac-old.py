import sys
sys.path.append('/home/yzhao4/new_repo_branchpoint/branch-point-prediction')
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from construct_graph import label_spreading, kevin_graph
from conformal_prediction import compute_score, prediction_set, find_qhat, solve_covariate_shift
from classifier import set_binary_label_rf, create_rf_classifiers, update_classifiers_trees
from classifier import train_and_predict_rf
from classifier import store_variable_importance, store_prediction_sets, store_prediction_sizes, store_test_probabilities
from scipy.stats import entropy
import logging
import time
import argparse
import scipy.sparse as sp
from utils import save_parser_info

parser = argparse.ArgumentParser(description='branchpoint prediction')

#data and i/o information
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--omit_tail', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--iter_graph', type=int, default=100)
parser.add_argument('--initial_trees', type=int, default=100)  # Initial number of trees
parser.add_argument('--trees_per_iteration', type=int, default=50)  # Trees to add per iteration

args = parser.parse_args()

save_parser_info(args, filename=args.output_dir + '/parser_arguments.txt')

start_time = time.time()

# Load data
data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Retina/ATAC/retina_peakvi_with_enhancers.h5ad')
data = data[~data.obs['BIP.type'].isin(['MuG', 'BIP']),:]

data.obs['terminal_state_cluster'] = data.obs['BIP.type'].astype('category')
terminal_states = ['RGC', 'CON', 'HRZ']
num_of_terminal_states = len(terminal_states)
new_categories = list(data.obs['terminal_state_cluster'].cat.categories) + ['TBD']
data.obs['terminal_state_cluster'] = data.obs['terminal_state_cluster'].cat.set_categories(new_categories)
# Perform the assignment safely
data.obs.loc[~data.obs['terminal_state_cluster'].isin(terminal_states), 'terminal_state_cluster'] = 'TBD'
pseudo_time_ranks = np.argsort(data.obs['palantir_pseudotime'][data.obs['terminal_state_cluster'] == 'TBD'].values)[::-1]

# Create dummy labels for the terminal states
dummy_labels = pd.get_dummies(data[data.obs['terminal_state_cluster'] != 'TBD'].obs['terminal_state_cluster']).astype('int')
# Get the mapping between variables in terminal_state_cluster and numbers
state_info_mapping = {state: idx for idx, state in enumerate(dummy_labels.columns)}
print("Mapping between terminal_state_cluster and numbers:", state_info_mapping)
available_types = dummy_labels.columns.tolist()

F = pd.DataFrame(np.zeros((data.n_obs, len(available_types))), 
                 index=data.obs_names.tolist(),
                 columns=available_types)
F.loc[data.obs_names[data.obs['terminal_state_cluster'] != 'TBD']] = dummy_labels.values
data.obsm['dummy_label'] = F

data.obs['visit'] = [1] * data.n_obs
data.obs.loc[data.obs_names[data.obs['terminal_state_cluster'] == 'TBD'], 'visit'] = 0

# Initialize conformal results dictionary
recruitment_size = round(np.sqrt(data.n_obs))
conformal_result = {
    'recruitment_size': recruitment_size, 
    'qhat': [], 
    'size': pd.DataFrame(), 
    'prediction_set': pd.DataFrame(),
    'prob_test': pd.DataFrame()
}

# Initialize iteration tracking in obs
data.obs['iteration_recruited'] = -1  # -1 for initial labeled cells, 0+ for iteration recruited
data.obs.loc[data.obs_names[data.obs['terminal_state_cluster'] != 'TBD'], 'iteration_recruited'] = -1

labeled_cells_barcode = data.obs_names[data.obs['visit'] == 1].tolist()
num_unlabeled_total = data[data.obs['terminal_state_cluster'] == 'TBD'].shape[0]

# Build the graph
#graph = kevin_graph(data.obsm['X_pca'], 10, metric="euclidean") 
graph = kevin_graph(data.obsm['X_peakvi'], 10, metric="euclidean") 

# Create Random Forest classifiers with initial number of trees
classifiers = create_rf_classifiers(len(available_types), n_estimators=args.initial_trees)

# Main loop
k = 0
cell_not_terminal = num_unlabeled_total 
cells2recruit_barcode = data.obs_names[data.obs['terminal_state_cluster'] == 'TBD'].tolist()

while np.sum(data.obs['visit']) < data.n_obs:
    print(f"The {k + 1}th round")

    if k > 0:
        classifiers = update_classifiers_trees(classifiers, args.trees_per_iteration)
        print(f"Updated classifiers to {classifiers[0].n_estimators} trees")

    recruitment_limit = np.min([recruitment_size * (k + 1), cell_not_terminal])
    test_ind = data[data.obs['terminal_state_cluster'] == 'TBD'][pseudo_time_ranks[
        k * recruitment_size:recruitment_limit]].obs_names.tolist()

    # we want to use cells close to test set in terms of pseudotime as calibration set
    train_pseudotime = data[labeled_cells_barcode].obs['palantir_pseudotime'].copy(deep=True)
    train_pseudotime = pd.DataFrame({'pseudotime':train_pseudotime})
    train_pseudotime_sorted = train_pseudotime.sort_values(by='pseudotime', ascending=True)
    cal_ind = train_pseudotime_sorted.index[:2 * recruitment_size].tolist()
    train_ind = train_pseudotime_sorted.index[2 * recruitment_size:].tolist()

    X_train = data[train_ind].X.copy()
    X_test = data[test_ind].X.copy()
    X_cal = data[cal_ind].X.copy()

    y_train = data[train_ind].obsm['dummy_label'].copy()
    y_cal = data[cal_ind].obsm['dummy_label'].copy()

    y_train_binary = set_binary_label_rf(y_train)

    trained_classifiers, prob_cal, prob_test = train_and_predict_rf(
        X_train, X_cal, X_test, cal_ind, test_ind, classifiers, y_train_binary
    )

    # Store variable importance for this iteration
    store_variable_importance(data, trained_classifiers, k, available_types, state_info_mapping, hvg=False)

    print(f"Iteration {k+1}: Using {trained_classifiers[0].n_estimators} trees")

    Fm = data.obsm['dummy_label'].copy(deep=True)
    Fm.loc[cal_ind] = prob_cal
    Fm.loc[test_ind] = prob_test

    F_true = data.obsm['dummy_label'].copy(deep=True)
    F_true.loc[cal_ind] = 0
    Fm = label_spreading(graph, Fm, alpha = args.alpha, iter_max=args.iter_graph)

    prob_cal_prop = Fm.loc[cal_ind]
    prob_test_prop = Fm.loc[test_ind]

    score = []
    rows, cols = np.nonzero(y_cal)
    nonzero_columns_by_row = [[] for _ in range(y_cal.shape[0])]
    for row, col in zip(rows, cols):
        nonzero_columns_by_row[row].append(col)

    for i in range(len(y_cal)):
        cumulative_sum = compute_score(prob_cal_prop.iloc[i, :], nonzero_columns_by_row[i])
        score.append(cumulative_sum)

    #do covariate shift
    calibration_features = prob_cal_prop.apply(lambda x: entropy(x), axis=1)
    test_features = prob_test_prop.apply(lambda x: entropy(x), axis=1)
    weights = solve_covariate_shift(calibration_features.values, test_features.values)

    df_score_weight = pd.DataFrame({'score': score, 'weight': weights})
    #order by score from small to large
    df_score_weight = df_score_weight.sort_values(by='score', ascending=True)
    qhat = find_qhat(df_score_weight['score'].tolist(), df_score_weight['weight'].tolist(), alpha = 0.1)

    conformal_result['qhat'].append(qhat)
    result = []
    num_pre_set = []
    for i in range(X_test.shape[0]):
        temp = prediction_set(prob_test_prop.iloc[i, :], qhat, omit_tail=args.omit_tail)
        num_pre_set.append(len(temp))
        result.append(temp)

    # Store sizes, prediction sets and test probabilities in conformal_result
    store_prediction_sizes(conformal_result, num_pre_set, test_ind)
    store_prediction_sets(conformal_result, result, test_ind, state_info_mapping)
    store_test_probabilities(conformal_result, prob_test_prop, test_ind, available_types)

    predicted_label = np.zeros((len(test_ind), len(available_types)))
    for row_index, cols in enumerate(result):
        for col_index in cols:
            predicted_label[row_index, col_index] = prob_test_prop.iloc[row_index, col_index]
    predicted_label = predicted_label / predicted_label.sum(axis=1, keepdims=True)
    positions = [data.obs_names.get_loc(elem) for elem in test_ind]
    data.obsm['dummy_label'].iloc[positions] = predicted_label
    
    # Mark which iteration these cells were recruited in
    data.obs.loc[test_ind, 'iteration_recruited'] = k
    
    k += 1

    data.obs.loc[test_ind, 'visit'] = 1
    labeled_cells_barcode = data.obs_names[data.obs['visit'] == 1].tolist()
    
    # Update classifiers for next iteration
    classifiers = trained_classifiers

data.write_h5ad(args.output_dir + '/data_complete_results.h5ad')
# Save conformal results
import pickle
with open(args.output_dir + '/conformal_result.pkl', 'wb') as f:
    pickle.dump(conformal_result, f)

logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds.")