import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
import pandas as pd
from scipy import sparse

def compute_local_sigma(data, knn):
    """
    Compute local adaptive sigma for each data point based on k-nearest neighbors.
    
    Parameters:
        data (ndarray): Input data points.
        knn (int): Number of nearest neighbors.
    
    Returns:
        local_sigma (ndarray): Local sigma values for each data point.
    """
    nbrs = NearestNeighbors(n_neighbors=knn+1).fit(data)
    distances, _ = nbrs.kneighbors(data)
    local_sigma = np.mean(distances[:, 1:], axis=1)  # Exclude self-distance
    return local_sigma


def kevin_graph(data, knn, metric="euclidean"):
    """
    Create a robust and connected MNN graph with local adaptive sigma.
    
    Parameters:
        data (ndarray): Input data points.
        knn (int): Number of nearest neighbors.
        metric (str): Distance metric for kNN.
    
    Returns:
        final_graph (csr_matrix): Sparse adjacency matrix of the connected graph.
    """
    # Step 1: Compute the kNN graph
    nbrs = NearestNeighbors(n_neighbors=knn, metric=metric).fit(data)
    adj = nbrs.kneighbors_graph(data, mode="distance")

    # Step 2: Compute the Minimum Spanning Tree (MST)
    mst = minimum_spanning_tree(adj)
    
    # Ensure MST is symmetric
    mst = mst.maximum(mst.T)

    # Step 3: Compute the mutual kNN graph
    adj2 = adj.minimum(adj.T)

    # Step 4: Compute local sigmas
    local_sigma = compute_local_sigma(data, knn)

    # Step 5: Apply adaptive Gaussian kernel to weights
    distances = adj2.toarray()
    weights = np.exp(-distances**2 / (local_sigma[:, None] * local_sigma[None, :]))
    weights[distances == 0] = 0  # Avoid self-loops
    weighted_adj = sp.csr_matrix(weights)

    # Step 6: Adjust MST weights
    # Find the minimum non-zero value in weighted_adj
    min_weight = weighted_adj.data[weighted_adj.data > 0].min()
    mst.data = np.full_like(mst.data, min_weight / 2)

    # Step 7: Combine MST and weighted kNN graph
    combined = sp.csr_matrix(np.maximum(weighted_adj.toarray(), mst.toarray()))

    return combined


def label_spreading(graph, F_mat, alpha=0.1, iter_max=100, tol=1e-6):
    """
    Optimized label propagation with sparse matrix support
    
    Parameters:
    -----------
    graph : scipy.sparse matrix or numpy array
        Adjacency matrix of the graph
    F_mat : pandas.DataFrame
        Initial label matrix
    self_impute : bool
        Whether to include self-loops (not used in current implementation)
    alpha : float
        Propagation parameter (0 < alpha < 1)
    iter_max : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    pandas.DataFrame
        Propagated label matrix
    """
    # Ensure graph dimensions match the length of F_mat
    assert graph.shape[0] == graph.shape[1] == F_mat.shape[0], "Graph and F_mat dimensions must match."
    
    # Convert graph to sparse format if it isn't already
    if not sp.issparse(graph):
        graph = sp.csr_matrix(graph)
    else:
        graph = graph.tocsr()  # Ensure CSR format for efficient operations
    
    # Initialize F with Y
    F_mat2 = F_mat.copy(deep=True)
    
    # Fill any NA with 0
    F_mat2 = F_mat2.fillna(0)
    
    # Convert to numpy array for computation
    F_initial = F_mat2.to_numpy()
    F_current = F_initial.copy()
    
    # Compute degree vector efficiently for sparse matrix
    degrees = np.array(graph.sum(axis=1)).flatten()
    
    # Handle isolated nodes (degree = 0)
    degrees_safe = np.where(degrees > 0, degrees, 1.0)
    
    # Compute D^(-1/2) as a sparse diagonal matrix
    D_inv_sqrt_diag = 1.0 / np.sqrt(degrees_safe)
    D_inv_sqrt = sp.diags(D_inv_sqrt_diag, format='csr')
    
    # Compute normalized adjacency matrix: D^(-1/2) * A * D^(-1/2)
    # This is more numerically stable than computing the Laplacian
    A_norm = D_inv_sqrt @ graph @ D_inv_sqrt
    
    # Pre-compute the constant term
    const_term = (1 - alpha) * F_initial
    
    # Iterative updates with convergence checking
    for iteration in range(iter_max):
        F_prev = F_current.copy()
        
        # Sparse matrix multiplication
        F_current = alpha * (A_norm @ F_current) + const_term
        
        # Check for convergence using relative change
        if iteration > 0:
            diff_norm = np.linalg.norm(F_current - F_prev, 'fro')
            f_norm = np.linalg.norm(F_current, 'fro')
            
            if f_norm > 0 and diff_norm / f_norm < tol:
                print(f"Converged after {iteration + 1} iterations")
                break
    
    # Normalize rows to make them probability distributions
    row_sums = F_current.sum(axis=1, keepdims=True)
    # Handle rows with zero sum (isolated nodes)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    F_current = F_current / row_sums
    
    # Convert back to DataFrame for consistent output format
    F_result = pd.DataFrame(F_current, index=F_mat.index, columns=F_mat.columns)
    
    return F_result


def label_propagation_truth_clamping(graph, F_mat, F_true, iter_max=100):
    """
    Optimized label propagation algorithm with truth clamping for sparse matrices.
    
    Parameters:
    -----------
    graph : scipy.sparse matrix
        The sparse affinity matrix representing the graph
    F_mat : pandas.DataFrame
        Matrix of initial label distributions
    F_true : pandas.DataFrame
        Matrix of ground truth labels (rows that sum to 1 are clamped)
    iter_max : int
        Maximum number of iterations
        
    Returns:
    --------
    numpy.ndarray
        Final label distributions after propagation

    Notes
    -----
    References: https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html
    """
    # Check dimensions
    assert graph.shape[0] == graph.shape[1] == F_mat.shape[0], "Graph and F_mat dimensions must match."
    
    # Convert graph to sparse if it isn't already
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(graph)
    
    # Normalize by row more efficiently for sparse matrices
    if sparse.issparse(graph):
        # Convert to CSR format for efficient row operations
        graph = graph.tocsr()
        row_sums = np.array(graph.sum(axis=1)).flatten()
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        # Efficient way to normalize a sparse matrix by row
        row_diag = sparse.diags(1.0 / row_sums)
        affinity_matrix = row_diag @ graph
    else:
        normalizer = graph.sum(axis=1)
        normalizer[normalizer == 0] = 1  # Avoid division by zero
        affinity_matrix = graph / normalizer[:, np.newaxis]
    
    # Create mask for fixed labels more efficiently
    F_true_np = F_true.fillna(0).to_numpy()
    row_sums_true = np.sum(F_true_np, axis=1)
    fixed_label_mask = np.isclose(row_sums_true, 1.0)
    F_true_np_fixed = F_true_np[fixed_label_mask, :]
    fixed_indices = np.where(fixed_label_mask)[0]
    
    # Initialize F_mat2 - convert once outside the loop
    F_mat2 = F_mat.fillna(0).to_numpy()
    
    # Iterative propagation
    for _ in range(iter_max):
        # Sparse matrix multiplication
        F_mat2 = affinity_matrix @ F_mat2
        
        # Normalize rows to sum to 1
        row_sums = np.sum(F_mat2, axis=1, keepdims=True)
        non_zero_rows = (row_sums > 0).flatten()
        F_mat2[non_zero_rows] = F_mat2[non_zero_rows] / row_sums[non_zero_rows]
        
        # Perform hard clamping - more efficient indexing
        F_mat2[fixed_indices] = F_true_np_fixed
    
    # Convert back to DataFrame for consistency

    F_mat2 = pd.DataFrame(F_mat2, index=F_mat.index)
    return F_mat2







































































def label_spreading_old(graph, F_mat, alpha=0.1, iter_max=100):
    """
    do label propagation
    restore the label for all cells before this iteration
    """
    # Ensure graph dimensions match the length of label_vec
    assert graph.shape[0] == graph.shape[1] == F_mat.shape[0], "Graph and F_mat dimensions must match."

    # Initialize F with Y
    F_mat2 = F_mat.copy(deep=True)

    # Fill any NA
    F_mat2 = F_mat2.fillna(0)

    # Convert to numpy array for computation
    F_mat2 = F_mat2.to_numpy()

    # Compute normalized Laplacian L
    diag_vec = np.array(graph.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(diag_vec))
    L_mat = D_inv_sqrt @ graph @ D_inv_sqrt

    # Iteratively update F with hard clamping
    for _ in range(iter_max):
        F_mat2 = alpha * L_mat @ F_mat2 + (1 - alpha) * F_mat2 # this line is a bug
        
    # Normalize rows of F_mat
    row_sums = F_mat2.sum(axis=1).reshape(-1, 1)
    F_mat2 /= row_sums

    # Convert back to DataFrame for consistent output format
    F_mat2 = pd.DataFrame(F_mat2, index=F_mat.index)

    return F_mat2


def truth_label_clamping(graph, F_mat, F_true, iter_max=100):
    """
    do clamping
    F_true means labels for cells at terminal states
    each time, we only restore labels for cells at terminal states
    """
    assert graph.shape[0] == graph.shape[1] == F_mat.shape[0], "Graph and F_mat dimensions must match."

    # Initialize F with Y
    F_mat2 = F_mat.copy(deep=True).fillna(0).to_numpy() #(129929, 8)

    # Compute normalized Laplacian L
    diag_vec = np.array(graph.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(diag_vec))
    L_mat = D_inv_sqrt @ graph @ D_inv_sqrt

    # Create a mask where rows in F_true sum to 1
    row_sums_true = F_true.sum(axis=1).to_numpy().reshape(-1, 1)
    fixed_label_mask = np.isclose(row_sums_true, 1.0)  # Ensure mask selects correct rows

    # Iteratively update F with hard clamping
    for _ in range(iter_max):
        F_mat2 = L_mat @ F_mat2
        # Perform hard clamping only for rows where F_true is a valid probability distribution
        F_mat2[fixed_label_mask.ravel(), :] = F_true.to_numpy()[fixed_label_mask.ravel(), :]
        #F_mat2[fixed_label_mask] = F_true.to_numpy()[fixed_label_mask]

    # Normalize rows of F_mat to ensure sum = 1
    row_sums = F_mat2.sum(axis=1, keepdims=True)
    nonzero_rows = row_sums != 0
    F_mat2[nonzero_rows.ravel()] /= row_sums[nonzero_rows.ravel()]

    # Convert back to DataFrame for consistency

    F_mat2 = pd.DataFrame(F_mat2, index=F_mat.index)

    return F_mat2


def label_propagation_clamping(graph, F_mat, self_impute=False, alpha=0.99, iter_max=100):
    """
    do label propagation
    restore the label for all cells before this iteration
    """
    # Ensure graph dimensions match the length of label_vec
    assert graph.shape[0] == graph.shape[1] == F_mat.shape[0], "Graph and F_mat dimensions must match."

    # Initialize F with Y
    F_mat2 = F_mat.copy(deep=True)

    # Fill any NA
    F_mat2 = F_mat2.fillna(0)

    # Convert to numpy array for computation
    F_mat2 = F_mat2.to_numpy()

    # Compute normalized Laplacian L
    diag_vec = np.array(graph.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(diag_vec))
    L_mat = D_inv_sqrt @ graph @ D_inv_sqrt

    if self_impute:
        L_mat = L_mat + sp.eye(L_mat.shape[0])

    # Create a mask for the fixed labels (non-NA entries in the original F_mat)
    fixed_label_mask = ~F_mat.isna().to_numpy()

    # Iteratively update F with hard clamping
    for _ in range(iter_max):
        F_mat2 = alpha * L_mat @ F_mat2 + (1 - alpha) * F_mat2

        # Perform hard clamping: reset fixed label values to their original values
        F_mat2[fixed_label_mask] = F_mat.to_numpy()[fixed_label_mask]

    # Normalize rows of F_mat
    row_sums = F_mat2.sum(axis=1).reshape(-1, 1)
    F_mat2 /= row_sums

    # Convert back to DataFrame for consistent output format
    F_mat2 = pd.DataFrame(F_mat2, index=F_mat.index)

    return F_mat2


import torch
def truth_label_clamping_gpu(graph, F_mat, F_true, iter_max=100, device='cuda'):
    """
    GPU-accelerated version using PyTorch.
    graph: scipy.sparse CSR matrix (affinity matrix)
    F_mat, F_true: pandas DataFrames
    """
    assert graph.shape[0] == graph.shape[1] == F_mat.shape[0], "Graph and F_mat dimensions must match."

    # Move data to torch tensors on GPU
    F_mat2 = torch.tensor(F_mat.fillna(0).to_numpy(), dtype=torch.float32, device=device)
    F_true_tensor = torch.tensor(F_true.to_numpy(), dtype=torch.float32, device=device)

    # Normalize graph to get L = D^(-1/2) * A * D^(-1/2)
    diag_vec = np.array(graph.sum(axis=1)).flatten()
    D_inv_sqrt = 1.0 / np.sqrt(diag_vec)
    D_inv_sqrt_mat = sp.diags(D_inv_sqrt)

    L = D_inv_sqrt_mat @ graph @ D_inv_sqrt_mat  # sparse normalized Laplacian
    L_coo = L.tocoo()
    indices = torch.tensor([L_coo.row, L_coo.col], dtype=torch.long, device=device)
    values = torch.tensor(L_coo.data, dtype=torch.float32, device=device)
    L_tensor = torch.sparse_coo_tensor(indices, values, size=L.shape, device=device)

    # Identify terminal states
    row_sums_true = F_true.sum(axis=1).to_numpy().reshape(-1, 1)
    fixed_label_mask = torch.isclose(
        torch.tensor(row_sums_true, dtype=torch.float32, device=device),
        torch.tensor(1.0, dtype=torch.float32, device=device)
    ).squeeze()

    # Iterative label propagation with clamping
    for _ in range(iter_max):
        F_mat2 = torch.sparse.mm(L_tensor, F_mat2)
        F_mat2[fixed_label_mask] = F_true_tensor[fixed_label_mask]

    # Normalize rows
    row_sums = F_mat2.sum(dim=1, keepdim=True).reshape(-1, 1)
    nonzero_mask = row_sums != 0
    F_mat2[nonzero_mask.ravel()] /= row_sums[nonzero_mask.ravel()]

    # Move back to CPU and return as DataFrame
    return pd.DataFrame(F_mat2.cpu().numpy(), index=F_mat.index)















