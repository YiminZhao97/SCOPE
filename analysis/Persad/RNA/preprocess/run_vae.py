import sys
sys.path.append('/home/yzhao4/new_repo_branchpoint/branch-point-prediction')
import scanpy as sc
import scanpy.external as sce
#import palantir
import anndata as ad
import os
from dataset import scdata
from model import VAE, train
import torch
import torch.nn as nn
import torch.optim as optim
import gc
import numpy as np
import pandas as pd
import time
from torch.utils.data import Subset, DataLoader, TensorDataset
import pickle

#load data
data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_seacell_pipeline/cd34_multiome_rna_palantir_without_cDC_CLP.h5ad')

# Subset the count matrix using the highly variable genes
X_hvg = data.layers['counts'][:, data.var.highly_variable]

# I want to make sure GATA1 GATA2 CEBPA SPI1 are in hvg

# Create a new AnnData object for HVGs
data.hvg = ad.AnnData(X=X_hvg, obs=data.obs.copy(), var=data.var[data.var.highly_variable].copy())
# Add library size to the new AnnData object
data.hvg.obs['library_size'] = np.squeeze(np.asarray(data.hvg.X.sum(axis=1)))

dataset = scdata(data.hvg)
dataset.info()

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#initialize model
dims = [dataset.n_genes, dataset.n_genes, 10, [128, 128], [128, 128]]
model = VAE(dims).to(device)
print(model)

#generate dataloader
num_train = int(len(dataset) * 0.9)
num_valid = len(dataset) - num_train
train_data = Subset(dataset, range(0, num_train))
valid_data = Subset(dataset, range(num_train, len(dataset)))
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 128, drop_last = False)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size = 128, drop_last = False)
all_data = torch.utils.data.DataLoader(dataset, batch_size = 128, drop_last = False)

t0 = time.time()
#model.load_state_dict(torch.load('/home/yzhao4/new_repo_branchpoint/Data/Palantir_bone_marrow/run_vae/checkpoint.pt'))
train(model, trainloader = train_loader, validloader = valid_loader, device = device, patience = 20, savepath = '/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell')
#print("Finished training takes {:.2f} min".format((time.time()-t0)/60))

#get denoised hvgs 

imputed_hvg = model.get_RNA_imputation(all_data, device = device)
data.obsm['imputed_hvg'] = imputed_hvg
#np.save('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data/denoised_hvg.npy', imputed_hvg)

#get latent space
latent_space = model.get_latent(all_data, device = device)
data.obsm['vae_latent_space'] = latent_space
#print(latent_space.shape)
#np.save('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data/vae_latent_space.npy', latent_space)

print(data)

data.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Seacell_HSPC/data_process_pipeline_basedon_seacell/cd34_multiome_rna_palantir_without_cDC_CLP_vae.h5ad')