import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/home/yzhao4/new_repo_branchpoint/Data/Simulation/Principle_curve/sim_different_skeleton/sim2/dat_sim2.csv', index_col=0)
pseudotime = pd.read_csv('/home/yzhao4/new_repo_branchpoint/Data/Simulation/Principle_curve/sim_different_skeleton/sim2/principle_curve_pseudotime_sim2.csv')

data = ad.AnnData(data)
data.obs['pseudotime'] = pseudotime['pseudotime'].tolist() 

#do clustering
sc.pp.highly_variable_genes(data)
sc.tl.pca(data, svd_solver='arpack')
sc.pp.neighbors(data, n_neighbors=10, n_pcs=40) #this step failed locally, run code on Hutch cluster
sc.tl.leiden(data, resolution=0.3)

#cluster 4, 5, 10 will be terminal states
df1 = pd.DataFrame({'pc1': data.obsm['X_pca'][:,0], 'pc2': data.obsm['X_pca'][:,1], 'Labels': data.obs['leiden'].to_list()})
sns.scatterplot(x="pc1", y="pc2", data=df1[df1['Labels'] != 'TBD'], hue='Labels')
plt.legend(loc='lower right', fontsize=10)
plt.title('PCA leiden clustering')
plt.savefig('/home/yzhao4/new_repo_branchpoint/Data/Simulation/Principle_curve/sim_different_skeleton/sim2/leiden_clustering_sim2.png', dpi = 400)
plt.clf()  
plt.close()  

terminal_state = ['4', '5', '10']

data.obs['leiden'] = data.obs['leiden'].astype('str')
data.obs.loc[((~data.obs['leiden'].isin(terminal_state)) | (data.obs['pseudotime'] <= 0.80)), 'leiden'] = 'TBD'

# Rename a column (e.g., 'old_name' to 'new_name')
data.obs.rename(columns={'leiden': 'cluster'}, inplace=True)

data.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Simulation/Principle_curve/sim_different_skeleton/sim2/simulation_processed_data_sim2.h5ad')


data = ad.read_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Simulation/Principle_curve/sim_different_skeleton/sim2/simulation_processed_data_sim2.h5ad')

# Convert cluster labels to segment labels
cluster_to_segment = {
    '4': 'segment1',
    '5': 'segment4',
    '10': 'segment3'
}

data.obs['cluster'] = data.obs['cluster'].map(cluster_to_segment).fillna(data.obs['cluster'])

# Save the updated data
data.write_h5ad('/home/yzhao4/new_repo_branchpoint/Data/Simulation/Principle_curve/sim_different_skeleton/sim2/simulation_processed_data_sim2_segment.h5ad')