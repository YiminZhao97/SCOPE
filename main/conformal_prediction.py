import numpy as np
import warnings
import pdb

def score_min_prob(prob, true_labels):
    """
    sum the ordered softmax output(high to low) only for true labels
    true_labels: list variable
    """
    #pdb.set_trace()
    prob_labels = prob[true_labels]
    return np.min(prob_labels)

def prediction_set_min_prob(prob, q):
    prediction_set = np.where(prob >= q)[0]
    return prediction_set

###################################################
def compute_score(prob, true_labels):
    """
    adaptive conformal score
    sum the ordered softmax output(high to low) until we cover all labels
    true_labels: list variable
    """
    #pdb.set_trace()
    sorted_indices = np.argsort(prob)[::-1]
    sorted_probs = prob[sorted_indices]
    cumulative_sum = 0
    covered_labels = set()
    for index, prob in zip(sorted_indices, sorted_probs):
        cumulative_sum += prob
        covered_labels.add(index)  
        if covered_labels.issuperset(true_labels):
            break
    return cumulative_sum

import numpy as np
import warnings

def prediction_set(prob, q, omit_tail=0):
    """
    prob: softmax output (convert to NumPy array)
    q: q hat determined by calibration set
    get prediction set for each test data point
    """
    # Ensure prob is a NumPy array
    prob = np.array(prob)

    sorted_indices = np.argsort(prob)[::-1]
    sorted_probs = prob[sorted_indices]
    cumulative_sum = 0
    prediction_set = []

    for idx, value in enumerate(sorted_probs):
        cumulative_sum += value
        if cumulative_sum + 1e-5 >= q:  # Ensure sum meets or exceeds q
            # Avoid out-of-bounds indexing
            safe_idx = min(idx + 1, len(sorted_indices))

            # Filter out indices where sorted_probs[i] >= omit_tail
            filtered_indices = [
                index for i, index in enumerate(sorted_indices[:safe_idx]) 
                if sorted_probs[i] >= omit_tail  # Use direct NumPy indexing
            ]
            #print(filtered_indices)  # Debugging output
            prediction_set.append(filtered_indices)
            break  # Stop once threshold is met

    prediction_set = [item for sublist in prediction_set for item in sublist]

    # Warning if the prediction set is empty
    if not prediction_set:
        warnings.warn("Warning: Prediction set is empty. Consider adjusting `q` or `omit_tail`.", UserWarning)
    
    return prediction_set



###################################################
def compute_score_onlytrue(prob, true_labels):
    """
    sum the ordered softmax output(high to low) only for true labels
    true_labels: list variable
    """
    #pdb.set_trace()
    sorted_indices = np.argsort(prob)[::-1]
    sorted_probs = prob[sorted_indices]
    cumulative_sum = 0
    for index, prob in zip(sorted_indices, sorted_probs):
        if index in true_labels:
            cumulative_sum += prob
    return cumulative_sum


def generate_descending_weights(n, r):
    """
    Generate descending weights that sum to 1.
    Parameters:
    n (int): Number of weights.
    r (float): Ratio between successive weights (0 < r < 1).

    Returns:
    list: A list of descending weights and should sum to 1.
    """
    if not (0 < r < 1):
        raise ValueError("Parameter 'r' must be in the range (0, 1).")

    # Compute the normalization factor (sum of geometric series)
    normalization_factor = (1 - r ** n) / (1 - r)

    # Generate weights
    weights = [(r ** i) / normalization_factor for i in range(n)]

    return weights


def find_qhat(score, weights, alpha=0.1):
    """
    Find the quantile estimate qhat using weighted cumulative sum.

    Parameters:
    - score: list or np.array of scores.
    - weights: list or np.array of weights corresponding to the scores.
    - alpha: float, significance level (default 0.1).

    Returns:
    - qhat: the smallest value in score such that cumulative weight exceeds alpha.
    """
    # Sort scores and weights together by score
    sorted_indices = np.argsort(score)
    sorted_score = np.array(score)[sorted_indices]
    sorted_weights = np.array(weights)[sorted_indices]

    # Compute cumulative sum of weights
    cumulative_weights = np.cumsum(sorted_weights)

    # Find the smallest score where cumulative weight exceeds alpha
    qhat_index = np.searchsorted(cumulative_weights, alpha, side='left')

    return sorted_score[qhat_index]

###########################################################################
import cvxpy as cp
import pandas as pd
def solve_covariate_shift(calibration_features, test_features, prob_cal_prop=None):
    """
    Solve the maximum entropy reweighting problem for covariate shift correction.

    Parameters:
    -----------
    calibration_features : numpy.ndarray
        Array of shape (n1, d) containing f(x_C_i) for each calibration point
    test_features : numpy.ndarray
        Array of shape (n2, d) containing f(x_T_i) for each test point
    prob_cal_prop : pandas.DataFrame, optional
        Calibration probabilities for fallback weighting scheme when optimization fails

    Returns:
    --------
    weights : numpy.ndarray
        Optimal weights for calibration points
    """
    n1 = calibration_features.shape[0]  # Number of calibration samples

    # Define the optimization variable (weights)
    w = cp.Variable(n1)

    # Objective: maximize entropy (-sum(w_i * log(w_i)))
    # Note: cp.entr(w) gives w_i * log(w_i), so we maximize the sum of entropy terms
    objective = cp.Maximize(cp.sum(cp.entr(w)))

    # Calculate mean features of test set
    test_mean = np.mean(test_features, axis=0)

    # Constraints

    epsilon = 1e-1  # Small tolerance
    constraints = [
    # Relaxed feature matching constraint with upper and lower bounds
    cp.sum(cp.multiply(w, calibration_features), axis=0) <= test_mean + epsilon,
    cp.sum(cp.multiply(w, calibration_features), axis=0) >= test_mean - epsilon,
    # Probability simplex constraints
    w >= 0,
    cp.sum(w) == 1
]
    # Formulate and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver = 'SCS')

    # Check if the problem was successfully solved
    if problem.status != 'optimal':
        print(f"Warning: Problem status is {problem.status}, not 'optimal'. Using fallback weighting scheme.")

        # Fallback: use max commitment-based weighting
        if prob_cal_prop is not None:
            max_commit = np.max(prob_cal_prop, axis=1)
            order_index = np.argsort(max_commit)
            ordered_max_commit = max_commit[order_index]
            weight = generate_descending_weights(n1, 0.9)
            ordered_max_commit_df = pd.DataFrame({'max_commit': ordered_max_commit})
            ordered_max_commit_df['weight'] = weight
            original_order_index = np.argsort(order_index)
            reordered_max_commit_df = ordered_max_commit_df.iloc[original_order_index].reset_index(drop=True)
            return reordered_max_commit_df['weight'].values
        else:
            # If prob_cal_prop not provided, use uniform weights as last resort
            print("Warning: prob_cal_prop not provided for fallback. Using uniform weights.")
            return np.ones(n1) / n1

    return w.value