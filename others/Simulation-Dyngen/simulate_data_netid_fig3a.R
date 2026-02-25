library(tidyverse)
library(dyngen)
library(anndata)

######## NETID Figure 3 simulation ############## 
set.seed(1)
backbone <- backbone_bifurcating()
config <- 
    initialise_model(
        backbone = backbone,
        num_cells = 4000,
        num_tfs = 50,
        download_cache_dir = tools::R_user_dir("dyngen","data"),
        num_targets = 200,
        num_hks = 50,
        verbose = FALSE
    )

out <- generate_dataset(
    config,
    format = "anndata",
    make_plots = TRUE
)


######## Figure 3 simulation ############## 
set.seed(1)
backbone <- backbone_bifurcating()
config <-
  initialise_model(
    backbone = backbone,
    num_cells = 1000,
    num_tfs = nrow(backbone$module_info),
    num_targets = 50,
    num_hks = 50,
    verbose = FALSE,
    download_cache_dir = tools::R_user_dir("dyngen", "data"),
    simulation_params = simulation_default(
      total_time = 1000,
      census_interval = 2, 
      ssa_algorithm = ssa_etl(tau = 300/3600),
      experiment_params = simulation_type_wild_type(num_simulations = 10)
    )
  )

out <- generate_dataset(
  config,
  format = "anndata",
  output_dir = '/home/yzhao4/new_repo_branchpoint/branch-point-prediction/Simulation/dyngen/tuto.h5ad',
  make_plots = TRUE
)

#print(out)
#saveRDS(out, "/home/yzhao4/new_repo_branchpoint/branch-point-prediction/Simulation/dyngen/dyngen_simulated_data_fig3a.rds")