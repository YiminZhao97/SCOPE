import sys
import time
import numpy as np
import csv
import gzip
import os
import scipy.io
import codecs
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
import scanpy as sc
import pdb
import gc
from anndata import AnnData
import anndata as ad
from scipy import sparse
sys.path.append('/Users/zhaoyimin/Desktop/Kevin/branch-point-prediction')

class scdata(Dataset):
    """
    Input: anndata for scRNA-seq
    """
    def __init__(self, anndata_rna):
        self.barcode = anndata_rna.obs_names.tolist()
        
        #self.RNA = np.array(anndata_rna.raw.to_adata().X.todense().T)
        #self.RNA = anndata_rna.raw.to_adata().X.T #after that, it is gene * cell
        self.barcode = anndata_rna.obs_names
        self.genes = anndata_rna.var_names
        #self.RNA = anndata_rna.raw.to_adata().X.todense() 
        if anndata_rna.raw is not None:
            self.RNA = anndata_rna.raw.X
            self.n_cells = anndata_rna.raw.to_adata().X.shape[0] #anndata_rna.n_obs
            self.n_genes = anndata_rna.raw.to_adata().X.shape[1] #anndata_rna.n_vars
        else:
            self.RNA = anndata_rna.X.todense()
            self.n_cells = anndata_rna.X.shape[0] #anndata_rna.n_obs
            self.n_genes = anndata_rna.X.shape[1] #anndata_rna.n_vars
        #self.RNA = anndata_rna.raw.X
        #self.library_size_rna = np.array(anndata_rna.obs['library size']).reshape((1,anndata_rna.n_obs)).astype(int)
        self.library_size_rna = np.array(anndata_rna.obs['library_size']).reshape((anndata_rna.n_obs, 1)).astype(int)
        #self.library_size_rna = np.array(anndata_rna.obs['library size']).flatten().astype(int)

    def __len__(self):
        #return self.RNA.shape[1]
        return self.RNA.shape[0]

    def __getitem__(self, index):
        #return self.data.getcol(index).toarray().squeeze()
        return self.library_size_rna[index,:], self.RNA[index,:]
        #return self.library_size_rna[index], self.RNA[index,:]
        #return self.library_size_rna[:,index], self.RNA[:,index]

    def info(self):
        print("\n===========================")
        print("Dataset Info")
        print('Cell number: {}\nGene number: {}'.format(self.n_cells, self.n_genes))
        print('===========================\n')        

