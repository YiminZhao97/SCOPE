from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import warnings

def set_binary_label_rf(y):
    """Convert multi-class labels to binary labels for each classifier."""
    return [
        y.iloc[:, i].values.astype(int) for i in range(y.shape[1])
    ]

def combine_pred_prob(*probs, cell_name):
    combined = np.column_stack([prob[:, 1] for prob in probs])
    row_sums = combined.sum(axis=1, keepdims=True)
    normalized_combined = combined / row_sums
    normalized_combined = pd.DataFrame(normalized_combined, index=cell_name)
    return normalized_combined

"""
functions related to train random forest
"""

def create_rf_classifiers(num_classifiers, n_estimators=100, class_weight='balanced', random_state=42, warm_start=True):
    """
    Create a list of Random Forest classifiers with warm_start enabled.

    Parameters:
    - num_classifiers (int): Number of classifiers to create.
    - n_estimators (int): Initial number of trees in each Random Forest.
    - class_weight (str or dict): Weighting of classes ('balanced' recommended).
    - random_state (int): Random seed for reproducibility.
    - warm_start (bool): Whether to use warm start for incremental learning.

    Returns:
    - List of RandomForestClassifier instances.
    """
    return [
        RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_state,
            warm_start=warm_start
        )
        for _ in range(num_classifiers)
    ]

def update_classifiers_trees(classifiers, trees_to_add):
    """
    Update the number of trees in each classifier using warm_start.
    """
    for clf in classifiers:
        clf.n_estimators += trees_to_add
    return classifiers

def train_rf_classifier(clf, X_train, y_train_binary):
    """Train the Random Forest classifier."""
    y_train_binary = np.array(y_train_binary).flatten()  # Ensure 1D format
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
                              message='.*class_weight presets.*warm_start.*',
                              category=UserWarning)
        clf.fit(X_train, y_train_binary)
    return clf

def predict_rf_classifier(clf, X):
    """Predict probabilities using the trained classifier."""
    return clf.predict_proba(X)

def get_variable_importance(classifiers):
    """
    Extract variable importance from all classifiers.
    
    Returns:
    - Dictionary with importance scores for each classifier
    """
    importance_dict = {}
    for i, clf in enumerate(classifiers):
        importance_dict[f'classifier_{i}'] = clf.feature_importances_
    return importance_dict

def train_and_predict_rf(X_train, X_cal, X_test, cal_ind, test_ind, classifiers, y_train_binary):
    """Train and predict in parallel."""
    # Train classifiers in parallel
    trained_classifiers = Parallel(n_jobs=-1, backend="loky")(
        delayed(train_rf_classifier)(clf, X_train, y_bin)
        for clf, y_bin in zip(classifiers, y_train_binary)
    )
    
    # Predict calibration probabilities in parallel
    prob_cal = Parallel(n_jobs=-1, backend="loky")(
        delayed(predict_rf_classifier)(clf, X_cal) for clf in trained_classifiers
    )
    
    # Predict test probabilities in parallel
    prob_test = Parallel(n_jobs=-1, backend="loky")(
        delayed(predict_rf_classifier)(clf, X_test) for clf in trained_classifiers
    )

    # combine the predicted probabilities
    prob_cal = combine_pred_prob(*prob_cal, cell_name=cal_ind)
    prob_test = combine_pred_prob(*prob_test, cell_name=test_ind)
    
    return trained_classifiers, prob_cal, prob_test

"""
variable importance
"""
def store_variable_importance(data, trained_classifiers, iteration, available_types, state_info_mapping, hvg=True):
    # Get number of features actually used in training (HVGs)
    n_features = trained_classifiers[0].n_features_in_
    importance_matrix = np.zeros((len(available_types), n_features))

    for i, clf in enumerate(trained_classifiers):
        if hasattr(clf, 'feature_importances_'):
            importance_matrix[i, :] = clf.feature_importances_

    # Create column names from state_info_mapping (sorted by index)
    column_names = [None] * len(available_types)
    for state, idx in state_info_mapping.items():
        column_names[idx] = state
    
    if hvg:
        # Create full-sized matrix and fill only HVG positions
        varm_key = f'feature_importance_hvg_iter_{iteration}'
        full_importance = np.zeros((data.n_vars, len(available_types)))

        # Map HVG importances back to their positions in the full gene space
        hvg_mask = data.var['highly_variable'] if 'highly_variable' in data.var.columns else slice(None)
        full_importance[hvg_mask, :] = importance_matrix.T

        # Verify dimensions are consistent
        assert full_importance.shape == (data.n_vars, len(available_types)), \
            f"Shape mismatch: expected {(data.n_vars, len(available_types))}, got {full_importance.shape}"

        data.varm[varm_key] = pd.DataFrame(
            full_importance,
            index=data.var_names,
            columns=column_names
        )
    else:
        varm_key = f'feature_importance_iter_{iteration}'

        # Verify dimensions are consistent
        assert importance_matrix.T.shape == (n_features, len(available_types)), \
            f"Shape mismatch: expected {(n_features, len(available_types))}, got {importance_matrix.T.shape}"

        data.varm[varm_key] = pd.DataFrame(
            importance_matrix.T,
            index=data.var_names,
            columns=column_names
        )

"""
predicted probabilities
"""

def store_test_probabilities(conformal_result, prob_test_prop, test_ind, available_types):
    """
    Store test probabilities as DataFrame in conformal_result with test_ind as row names.
    
    Parameters:
    -----------
    conformal_result : dict
        The conformal result dictionary to store results in
    prob_test_prop : DataFrame
        Test probabilities for the current iteration
    test_ind : list
        List of cell identifiers for the test set
    available_types : list
        List of available cell types/states
    """
    # Store test probabilities with test_ind as row names
    prob_test_df = pd.DataFrame(prob_test_prop.values, 
                               index=test_ind, 
                               columns=available_types)
    
    # Concatenate with previous test probabilities
    if conformal_result['prob_test'].empty:
        conformal_result['prob_test'] = prob_test_df
    else:
        conformal_result['prob_test'] = pd.concat([
            conformal_result['prob_test'], 
            prob_test_df
        ])


"""
prediction sets
"""
def store_prediction_sets(conformal_result, result, test_ind, state_info_mapping):
    """
    Convert prediction indices to sets of labels and store in conformal_result.
    
    Parameters:
    -----------
    conformal_result : dict
        The conformal result dictionary to store results in
    result : list
        List of prediction sets (each element is a list of indices)
    test_ind : list
        List of cell identifiers for the test set
    state_info_mapping : dict
        Mapping from state names to indices
    """
    # Convert prediction indices to sets of labels for each cell
    prediction_set_labels = []
    for row_index, cols in enumerate(result):
        labels = set()
        for col_index in cols:
            # Get the cell type name from state_info_mapping
            cell_type = [name for name, idx in state_info_mapping.items() if idx == col_index][0]
            labels.add(cell_type)
        prediction_set_labels.append(labels)
    
    # Create prediction set dataframe for this iteration
    iteration_prediction_set = pd.DataFrame({
        'prediction_set': prediction_set_labels
    })
    iteration_prediction_set.index = test_ind
    
    # Concatenate with previous prediction sets
    if conformal_result['prediction_set'].empty:
        conformal_result['prediction_set'] = iteration_prediction_set
    else:
        conformal_result['prediction_set'] = pd.concat([
            conformal_result['prediction_set'], 
            iteration_prediction_set
        ])


def store_prediction_sizes(conformal_result, num_pre_set, test_ind):
    """
    Store prediction set sizes as DataFrame in conformal_result.
    
    Parameters:
    -----------
    conformal_result : dict
        The conformal result dictionary to store results in
    num_pre_set : list
        List of prediction set sizes for each test cell
    test_ind : list
        List of cell identifiers for the test set
    """
    # Store sizes as DataFrame with cell mapping
    iteration_size_df = pd.DataFrame({
        'prediction_set_size': num_pre_set
    })
    iteration_size_df.index = test_ind
    
    # Concatenate with previous sizes
    if conformal_result['size'].empty:
        conformal_result['size'] = iteration_size_df
    else:
        conformal_result['size'] = pd.concat([
            conformal_result['size'], 
            iteration_size_df
        ])

"""
others
"""

def convert_prediction_indices_to_state_names(result, state_info_mapping):
    """
    Convert prediction set indices to state names using the state_info_mapping.

    Parameters:
    - result: List of prediction sets (indices).
    - state_info_mapping: Dictionary mapping state names to indices.

    Returns:
    - List of prediction sets with state names instead of indices.
    """
    result_named = []
    for pred_set in result:
        named_set = [list(state_info_mapping.keys())[list(state_info_mapping.values()).index(idx)] for idx in pred_set]
        result_named.append(named_set)
    return result_named

