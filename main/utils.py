from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
import numpy as np
import anndata as ad
import os
#import mellon
import palantir

#mpl.use('Agg')
"""
functions for plot
"""
def plot_tSNE(data, label, output_dir):
    """
    data: sample * feature
    """
    tsne = TSNE(n_components=2, random_state = 2023) 
    result = tsne.fit_transform(data)

    df = pd.DataFrame({'tSNE1':result[:,0], 'tSNE2':result[:,1],'Labels':label})

    sns.scatterplot(x="tSNE1", y="tSNE2", hue="Labels",data = df)
    plt.legend(loc = 'upper left', fontsize = 5)
    plt.title('tSNE plot')
    plt.savefig(output_dir, dpi = 400)

def plot_umap_rawdata(adata, output_dir):
    """
    Description: plot umap for count matrix, basically just follow scanpy's standard workflow
    adata: anndata object
    output_dir: output direction for UMAP plot
    """
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata, init_pos='paga')
    embedding = adata.obsm['X_umap']

    df = pd.DataFrame({'umap1':embedding[:,0], 'umap2':embedding[:,1],'Labels':adata.cluster.tolist()})
    sns.scatterplot(x="umap1", y="umap2", hue="Labels",data = df)
    plt.legend(loc = 'upper left', fontsize = 5)
    plt.title('UMAP')
    plt.savefig(output_dir, dpi = 400)

    return embedding

def plot_umap_latent_space(latent, annotations, output_dir):
    """
    Description: plot umap for some embeddings such as latent space of VAE
    treat embedding as pca embedding and then follow scanpy's workflow

    latent: latent embedding, numpy array
    annotation: cell type annotations, list
    output_dir: output direction, str
    """
    adata = ad.AnnData(X=latent)
    adata.obs_names = [f'cell{i}' for i in range(latent.shape[0])]
    adata.var_names = [f'gene{j}' for j in range(latent.shape[1])]

    adata.obsm['X_pca'] = latent
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=latent.shape[1])
    sc.tl.umap(adata, init_pos='paga')
    #sc.tl.umap(adata)
    embedding = adata.obsm['X_umap']

    df = pd.DataFrame({'umap1':embedding[:,0], 'umap2':embedding[:,1],'Labels':annotations})
    sns.scatterplot(x="umap1", y="umap2", hue="Labels",data = df)
    plt.legend(loc = 'upper left', fontsize = 5)
    plt.title('UMAP')
    plt.savefig(output_dir, dpi = 400)
    return 0

def gif_prediction_set(directory, savepath, duration):
    import imageio
    filenames = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    sorted_file_paths = sorted(filenames, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    images = []
    for filename in sorted_file_paths:
        images.append(imageio.imread(filename))
    imageio.mimsave(savepath, images, duration=duration)  

def barplot_size_pre_set(num_pre, save_path):
    counts = num_pre['num_pre_set'].value_counts().loc[[1, 2, 3]]
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    bar_plot = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title('Number of Occurrences of 1, 2, and 3')
    plt.xlabel('Number')
    plt.ylabel('Count')
    plt.savefig(save_path, dpi = 300)

def plot_cover_one_fate(df, fig_name, save_path = ''):
    plt.figure(figsize=(6, 4))
    #label_colors = {1: np.array([0.93126922, 0.82019218, 0.7971481 , 1.]),2: np.array([0.66265275, 0.40279894, 0.5599294 , 1.]),
    #3: np.array([0.17508656, 0.11840023, 0.24215989, 1.])}
    #label_colors = {1: np.array([0.93126922, 0.82019218, 0.7971481,  1.]),2: np.array([0.78404409, 0.52926605, 0.62005689, 1.]),3: np.array([0.5151069, 0.29801048, 0.49050619, 1. ]), 4: np.array([0.17508656, 0.11840023, 0.24215989, 1])}
    df_0 = df[~df['Labels'].isin([0, 1])]
    df_1 = df[df['Labels'] == 0]
    df_2 = df[df['Labels'] == 1]
    sns.scatterplot(x='umap1', y='umap2', color='grey', data=df_0, hue='Labels')
    sns.scatterplot(x='umap1', y='umap2', color='grey', data=df_1)
    sns.scatterplot(x='umap1', y='umap2', data=df_2, color='purple') 
    plt.legend(loc = 'upper right', fontsize = 10)
    plt.title(fig_name)
    #plt.show()
    plt.savefig(save_path, dpi=400)
    plt.clf()  
    plt.close()

"""
functions for recruiting
"""

def compute_nn(X, n):
    """
    Description: compute distance matrix and nearest neighbours based on embedding
    X: embedding
    n: number of neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances, indices

def recruit_cells(allcells, labeled_cell_index, unlabeled_cell_index, distances, k):
    """
    allcells: index of all cells eg. Index(['Run4_120703408880541', 'Run4_120703409056541'])
    labeled_cell_index: index of all labeled cells
    unlabeled_cell_index: index of all unlabeled cells
    distance: distance matrix output by nbrs.kneighbors
    k: number of cells to recruit
    """
    unlabeled_dist_matrix = distances[[allcells.get_loc(label) for label in unlabeled_cell_index], :]
    dis_with_labeled = unlabeled_dist_matrix[:,[allcells.get_loc(label) for label in labeled_cell_index]]
    avg_dis = np.mean(dis_with_labeled, axis=1)
    min_dis_indices = np.argsort(avg_dis)[:k]
    recruit_cell_index = unlabeled_cell_index[min_dis_indices]
    return recruit_cell_index

"""
functions for density calculation
"""
def compute_density(ad, n_components = 10):
    #dm_res = palantir.utils.run_diffusion_maps(ad, pca_key="X_pca", n_components=n_components)
    model = mellon.DensityEstimator()
    log_density = model.fit_predict(ad.obsm["DM_EigenVectors"])
    return model, log_density


"""
functions used for random forest
"""
def set_binary_label_rf(y):
    #convert m labels to m binary labels
    y_rf1 = (y[:, 0] > 0).astype(int)
    y_rf2 = (y[:, 1] > 0).astype(int)
    y_rf3 = (y[:, 2] > 0).astype(int)
    return y_rf1, y_rf2, y_rf3

def combine_pred_prob(prob1, prob2, prob3):
    #the first column corresponds to the probability of being 0 and the second column corresponds to the probability of being 1
    combined = np.column_stack((prob1[:,1], prob2[:,1], prob3[:,1]))

    #avoid sum = 0 (none of rf think the sample belongs to the class)
    for i in range(combined.shape[0]):
        if np.sum(combined[i, :]) == 0:
            combined[i, :] = 1 / combined.shape[1]

    row_sums = combined.sum(axis=1, keepdims=True)
    normalized_combined = combined / row_sums
    return normalized_combined




def save_parser_info(args, filename='parser_arguments.txt'):
    """Save parser argument information to a text file"""
    
    parser_info = f"""Branchpoint Prediction Script Arguments
========================================

--output_dir: {args.output_dir}
  Type: str
  Default: ''

--omit_tail: {args.omit_tail}
  Type: float
  Default: 0

--alpha: {args.alpha}
  Type: float
  Default: 0.1

--iter_graph: {args.iter_graph}
  Type: int
  Default: 100

--initial_trees: {args.initial_trees}
  Type: int
  Default: 100
  Description: Initial number of trees

--trees_per_iteration: {args.trees_per_iteration}
  Type: int
  Default: 50
  Description: Trees to add per iteration
"""
    
    with open(filename, 'w') as f:
        f.write(parser_info)
    
    print(f"Parser information saved to {filename}")