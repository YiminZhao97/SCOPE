"""
SCOPE: semi-supervised Conformal Prediction
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import entropy
import time
import pickle
from construct_graph import label_spreading, kevin_graph
from conformal_prediction import compute_score, prediction_set, find_qhat, solve_covariate_shift
from classifier import set_binary_label_rf, create_rf_classifiers, update_classifiers_trees
from classifier import train_and_predict_rf
from classifier import store_variable_importance, store_prediction_sets, store_prediction_sizes, store_test_probabilities

class SCOPE:
    """
    SCOPE: semi-supervised Conformal Prediction

    A class for predicting cell fate trajectories using conformal prediction
    and iterative semi-supervised learning.

    Parameters
    ----------
    data : AnnData
        Annotated data object with cells × features
        Required obs: 'terminal_state_cluster', 'palantir_pseudotime'
        Required obsm: latent space for graph (e.g., 'vae_latent_space', 'X_pca')
        Features: either data.X or data.obsm[feature_key]
    feature_key : str
        Key in data.obsm containing features for prediction (default: 'imputed_hvg')
        Ignored if use_X=True
    latent_key : str
        Key in data.obsm containing latent space for graph construction (default: 'vae_latent_space')
    alpha : float
        Label spreading parameter (default: 0.1)
    omit_tail : float
        Tail omission parameter for prediction sets (default: 0)
    iter_graph : int
        Number of label spreading iterations (default: 100)
    initial_trees : int
        Initial number of trees in random forest (default: 100)
    trees_per_iteration : int
        Trees to add per iteration (default: 50)
    n_neighbors : int
        Number of neighbors for graph construction (default: 10)
    recruitment_size : int, optional
        Number of cells to recruit per iteration (default: None, auto-computed as sqrt(n_obs))
    use_X : bool
        If True, use data.X as feature matrix instead of data.obsm[feature_key] (default: False)
        Useful when no imputation is available

    Examples
    --------
    >>> import anndata as ad
    >>> from scope import SCOPE
    >>>
    >>> # Load your data (with Palantir already run)
    >>> data = ad.read_h5ad('your_data.h5ad')
    >>>
    >>> # Create terminal_state_cluster column
    >>> # Mark known terminal states, set unknown cells to 'TBD'
    >>> data.obs['terminal_state_cluster'] = ...
    >>>
    >>> # Initialize SCOPE
    >>> scope = SCOPE(
    ...     data=data,
    ...     feature_key='imputed_hvg',
    ...     latent_key='vae_latent_space',
    ...     alpha=0.1,
    ...     iter_graph=100
    ... )
    >>>
    >>> # Run pipeline
    >>> scope.prepare_data() \\
    ...      .initialize_conformal_result() \\
    ...      .build_graph() \\
    ...      .initialize_classifiers() \\
    ...      .run_all() \\
    ...      .save_results('output_dir')
    """

    def __init__(self, data, feature_key='imputed_hvg', latent_key='vae_latent_space',
                 alpha=0.1, omit_tail=0, iter_graph=100, initial_trees=100,
                 trees_per_iteration=50, n_neighbors=10, recruitment_size=None, use_X=False):
        """Initialize SCOPE with data and parameters."""
        # Import dependencies first

        self.data = data
        self.feature_key = feature_key
        self.latent_key = latent_key
        self.alpha = alpha
        self.omit_tail = omit_tail
        self.iter_graph = iter_graph
        self.initial_trees = initial_trees
        self.trees_per_iteration = trees_per_iteration
        self.n_neighbors = n_neighbors
        self._recruitment_size_param = recruitment_size
        self.use_X = use_X

        # Validate input data
        self._validate_input()

        # To be initialized
        self.pseudo_time_ranks = None
        self.available_types = None
        self.state_info_mapping = None
        self.recruitment_size = None
        self.conformal_result = None
        self.graph = None
        self.classifiers = None
        self.iteration = 0

    def _validate_input(self):
        """Validate that required fields exist in data."""
        if 'terminal_state_cluster' not in self.data.obs.columns:
            raise ValueError("data.obs must contain 'terminal_state_cluster' column")
        if 'palantir_pseudotime' not in self.data.obs.columns:
            raise ValueError("data.obs must contain 'palantir_pseudotime' column")

        # Validate feature matrix location
        if not self.use_X:
            if self.feature_key not in self.data.obsm.keys():
                raise ValueError(f"data.obsm must contain '{self.feature_key}' (or set use_X=True)")
        else:
            if self.data.X is None:
                raise ValueError("data.X is None, cannot use use_X=True")

        # Validate latent space
        if self.latent_key not in self.data.obsm.keys():
            raise ValueError(f"data.obsm must contain '{self.latent_key}'")

    def prepare_data(self):
        """
        Prepare data for SCOPE analysis.

        This function:
        - Identifies cells marked as 'TBD' (To Be Determined) in terminal_state_cluster
        - Creates dummy labels for known terminal states
        - Sorts cells by pseudotime
        - Initializes visit tracking

        Returns
        -------
        self : SCOPE
            Returns self for method chaining
        """
        # Sort cells by pseudotime
        tbd_mask = self.data.obs['terminal_state_cluster'] == 'TBD'
        self.pseudo_time_ranks = np.argsort(
            self.data.obs['palantir_pseudotime'][tbd_mask].values
        )[::-1]

        # Create dummy labels for terminal states
        labeled_mask = self.data.obs['terminal_state_cluster'] != 'TBD'
        dummy_labels = pd.get_dummies(
            self.data.obs['terminal_state_cluster'][labeled_mask]
        ).astype('int')

        # Create state mapping
        self.state_info_mapping = {
            state: idx for idx, state in enumerate(dummy_labels.columns)
        }
        self.available_types = dummy_labels.columns.tolist()

        print("Mapping between terminal_state_cluster and numbers:", self.state_info_mapping)

        # Initialize dummy label matrix
        F = pd.DataFrame(
            np.zeros((self.data.n_obs, len(self.available_types))),
            index=self.data.obs_names.tolist(),
            columns=self.available_types
        )
        F.loc[self.data.obs_names[labeled_mask]] = dummy_labels.values
        self.data.obsm['dummy_label'] = F

        # Mark visited cells
        self.data.obs['visit'] = 0
        self.data.obs.loc[self.data.obs_names[labeled_mask], 'visit'] = 1

        # Initialize iteration tracking
        self.data.obs['iteration_recruited'] = -1
        self.data.obs.loc[self.data.obs_names[labeled_mask], 'iteration_recruited'] = -1

        print(f"Prepared data: {labeled_mask.sum()} labeled cells, {tbd_mask.sum()} unlabeled cells")

        return self

    def initialize_conformal_result(self):
        """
        Initialize conformal prediction result structures.

        Returns
        -------
        self : SCOPE
            Returns self for method chaining
        """
        # Use custom recruitment_size if provided, otherwise auto-compute
        if self._recruitment_size_param is not None:
            self.recruitment_size = self._recruitment_size_param
            print(f"Using custom recruitment size: {self.recruitment_size}")
        else:
            self.recruitment_size = round(np.sqrt(self.data.n_obs))
            print(f"Auto-computed recruitment size: {self.recruitment_size} (sqrt of {self.data.n_obs} cells)")

        self.conformal_result = {
            'recruitment_size': self.recruitment_size,
            'qhat': [],
            'size': pd.DataFrame(),
            'prediction_set': pd.DataFrame(),
            'prob_test': pd.DataFrame()
        }

        print(f"Initialized conformal results with recruitment size: {self.recruitment_size}")

        return self

    def build_graph(self):
        """
        Build k-nearest neighbor graph for label spreading.

        Returns
        -------
        self : SCOPE
            Returns self for method chaining
        """
        print(f"Building graph with {self.n_neighbors} neighbors...")
        self.graph = kevin_graph(
            self.data.obsm[self.latent_key],
            self.n_neighbors,
            metric="euclidean"
        )
        print("Graph construction complete")

        return self

    def initialize_classifiers(self):
        """
        Initialize random forest classifiers.

        Returns
        -------
        self : SCOPE
            Returns self for method chaining
        """
        self.classifiers = create_rf_classifiers(
            len(self.available_types),
            n_estimators=self.initial_trees
        )
        print(f"Initialized {len(self.available_types)} classifiers with {self.initial_trees} trees each")

        return self

    def _get_features(self, indices):
        """
        Get feature matrix for given cell indices.

        Parameters
        ----------
        indices : list
            Cell indices

        Returns
        -------
        features : np.ndarray or sparse matrix
            Feature matrix for the specified cells
        """
        if self.use_X:
            # Use data.X directly
            import scipy.sparse as sp
            X = self.data[indices].X
            if sp.issparse(X):
                return X.toarray().copy()
            else:
                return X.copy()
        else:
            # Use data.obsm[feature_key]
            return self.data[indices].obsm[self.feature_key].copy()

    def _get_train_cal_test_split(self, test_ind, labeled_cells_barcode):
        """Get train, calibration, and test indices."""
        train_pseudotime = self.data[labeled_cells_barcode].obs['palantir_pseudotime'].copy(deep=True)
        train_pseudotime = pd.DataFrame({'pseudotime': train_pseudotime})
        train_pseudotime_sorted = train_pseudotime.sort_values(by='pseudotime', ascending=True)

        cal_ind = train_pseudotime_sorted.index[:2 * self.recruitment_size].tolist()
        train_ind = train_pseudotime_sorted.index[2 * self.recruitment_size:].tolist()

        return train_ind, cal_ind, test_ind

    def _compute_conformal_scores(self, y_cal, prob_cal_prop):
        """Compute conformal prediction scores."""
        score = []
        rows, cols = np.nonzero(y_cal)
        nonzero_columns_by_row = [[] for _ in range(y_cal.shape[0])]
        for row, col in zip(rows, cols):
            nonzero_columns_by_row[row].append(col)

        for i in range(len(y_cal)):
            cumulative_sum = compute_score(prob_cal_prop.iloc[i, :], nonzero_columns_by_row[i])
            score.append(cumulative_sum)

        return score

    def _apply_covariate_shift(self, prob_cal_prop, prob_test_prop, score):
        """Apply covariate shift adjustment and compute qhat."""
        calibration_features = prob_cal_prop.apply(lambda x: entropy(x), axis=1)
        test_features = prob_test_prop.apply(lambda x: entropy(x), axis=1)
        weights = solve_covariate_shift(calibration_features.values, test_features.values, prob_cal_prop=prob_cal_prop)

        df_score_weight = pd.DataFrame({'score': score, 'weight': weights})
        df_score_weight = df_score_weight.sort_values(by='score', ascending=True)
        qhat = find_qhat(df_score_weight['score'].tolist(), df_score_weight['weight'].tolist(), alpha=0.1)

        return qhat

    def _generate_prediction_sets(self, prob_test_prop, qhat, test_ind):
        """Generate prediction sets for test cells."""
        result = []
        num_pre_set = []

        for i in range(prob_test_prop.shape[0]):
            temp = prediction_set(prob_test_prop.iloc[i, :], qhat, omit_tail=self.omit_tail)
            num_pre_set.append(len(temp))
            result.append(temp)

        # Store results
        store_prediction_sizes(self.conformal_result, num_pre_set, test_ind)
        store_prediction_sets(self.conformal_result, result, test_ind, self.state_info_mapping)
        store_test_probabilities(self.conformal_result, prob_test_prop, test_ind, self.available_types)

        return result

    def _update_labels(self, test_ind, result, prob_test_prop):
        """Update predicted labels for test cells."""
        predicted_label = np.zeros((len(test_ind), len(self.available_types)))
        for row_index, cols in enumerate(result):
            for col_index in cols:
                predicted_label[row_index, col_index] = prob_test_prop.iloc[row_index, col_index]

        predicted_label = predicted_label / predicted_label.sum(axis=1, keepdims=True)
        positions = [self.data.obs_names.get_loc(elem) for elem in test_ind]
        self.data.obsm['dummy_label'].iloc[positions] = predicted_label

        self.data.obs.loc[test_ind, 'iteration_recruited'] = self.iteration
        self.data.obs.loc[test_ind, 'visit'] = 1

    def run_scope(self):
        """
        Run one iteration of SCOPE prediction.

        This function performs:
        1. Updates classifiers (adds trees if not first iteration)
        2. Selects test cells based on pseudotime
        3. Trains random forest classifiers
        4. Applies label spreading on graph
        5. Computes conformal prediction sets
        6. Updates cell labels

        Returns
        -------
        bool
            True if iteration completed successfully, False if no more cells to process
        """
        # Check if there are cells left to process
        if np.sum(self.data.obs['visit']) >= self.data.n_obs:
            print("All cells have been processed")
            return False

        print(f"\n{'='*60}")
        print(f"SCOPE Iteration {self.iteration + 1}")
        print(f"{'='*60}")

        # Update classifiers (add trees after first iteration)
        if self.iteration > 0:
            self.classifiers = update_classifiers_trees(self.classifiers, self.trees_per_iteration)
            print(f"Updated classifiers to {self.classifiers[0].n_estimators} trees")

        # Get labeled cells
        labeled_cells_barcode = self.data.obs_names[self.data.obs['visit'] == 1].tolist()

        # Determine test set
        num_unlabeled_total = (self.data.obs['terminal_state_cluster'] == 'TBD').sum()
        recruitment_limit = np.min([self.recruitment_size * (self.iteration + 1), num_unlabeled_total])
        test_ind = self.data[self.data.obs['terminal_state_cluster'] == 'TBD'][
            self.pseudo_time_ranks[self.iteration * self.recruitment_size:recruitment_limit]
        ].obs_names.tolist()

        print(f"Processing {len(test_ind)} test cells (total labeled: {len(labeled_cells_barcode)})")

        # Split into train/cal/test
        train_ind, cal_ind, test_ind = self._get_train_cal_test_split(test_ind, labeled_cells_barcode)

        # Get features using appropriate method (either data.X or data.obsm[feature_key])
        X_train = self._get_features(train_ind)
        X_test = self._get_features(test_ind)
        X_cal = self._get_features(cal_ind)

        y_train = self.data[train_ind].obsm['dummy_label'].copy()
        y_cal = self.data[cal_ind].obsm['dummy_label'].copy()

        y_train_binary = set_binary_label_rf(y_train)

        # Train and predict
        print("Training classifiers and generating predictions...")
        trained_classifiers, prob_cal, prob_test = train_and_predict_rf(
            X_train, X_cal, X_test, cal_ind, test_ind, self.classifiers, y_train_binary
        )

        # Store variable importance
        store_variable_importance(
            self.data, trained_classifiers, self.iteration,
            self.available_types, self.state_info_mapping
        )

        # Label spreading
        print("Applying label spreading...")
        Fm = self.data.obsm['dummy_label'].copy(deep=True)
        Fm.loc[cal_ind] = prob_cal
        Fm.loc[test_ind] = prob_test

        F_true = self.data.obsm['dummy_label'].copy(deep=True)
        F_true.loc[cal_ind] = 0
        Fm = label_spreading(self.graph, Fm, alpha=self.alpha, iter_max=self.iter_graph)

        prob_cal_prop = Fm.loc[cal_ind]
        prob_test_prop = Fm.loc[test_ind]

        # Compute conformal scores
        print("Computing conformal prediction sets...")
        score = self._compute_conformal_scores(y_cal, prob_cal_prop)

        # Apply covariate shift adjustment
        qhat = self._apply_covariate_shift(prob_cal_prop, prob_test_prop, score)
        self.conformal_result['qhat'].append(qhat)
        print(f"Qhat: {qhat:.4f}")

        # Generate prediction sets
        result = self._generate_prediction_sets(prob_test_prop, qhat, test_ind)

        # Update labels
        self._update_labels(test_ind, result, prob_test_prop)

        # Update classifiers for next iteration
        self.classifiers = trained_classifiers

        # Increment iteration counter
        self.iteration += 1

        print(f"Iteration {self.iteration} complete")

        return True

    def run_all(self):
        """
        Run SCOPE until all cells are processed.

        Returns
        -------
        self : SCOPE
            Returns self for method chaining
        """
        start_time = time.time()

        while self.run_scope():
            pass

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"SCOPE complete: {self.iteration} iterations in {elapsed:.2f} seconds")
        print(f"{'='*60}")

        return self

    def save_results(self, output_dir):
        """
        Save SCOPE results to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save annotated data
        output_path = os.path.join(output_dir, 'data_complete_results.h5ad')
        self.data.write_h5ad(output_path)
        print(f"Saved data to {output_path}")

        # Save conformal results
        conformal_path = os.path.join(output_dir, 'conformal_result.pkl')
        with open(conformal_path, 'wb') as f:
            pickle.dump(self.conformal_result, f)
        print(f"Saved conformal results to {conformal_path}")
