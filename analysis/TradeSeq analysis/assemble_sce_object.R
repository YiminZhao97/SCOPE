library(Seurat)
library(tradeSeq)
library(RColorBrewer)
library(SingleCellExperiment)
library(slingshot)
library(zellkonverter)
library(Matrix)
library(readr)

setwd('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure4 Seacell Palantir/tradeseq new')


# Read matrix (sparse or dense)
expr_mat <- Matrix::readMM("./bp candidate1/assemble Palantir/expression_matrix.mtx")
expr_mat <- t(expr_mat)

# Read features and barcodes
genes <- read_csv("./bp candidate1/assemble Palantir/features.csv", col_names = FALSE)[[1]]
genes = genes[2:length(genes)]
cells <- read_csv("./bp candidate1/assemble Palantir/barcodes.csv", col_names = FALSE)[[1]]
cells = cells[2:length(cells)]

# Assign row and column names
rownames(expr_mat) <- genes
colnames(expr_mat) <- cells

# Read pseudotime
pseudotime <- read_csv("./bp candidate1/assemble Palantir/pseudotime.csv")
rownames(pseudotime) <- cells  # Make sure rownames match colnames of matrix

col_data <- DataFrame(
  pseudotime = pseudotime[[2]]
  #leiden = factor(leiden[[1]])
)
rownames(col_data) <- cells


#umap <- read_csv("./assemble Palantir/X_umap.csv")
#rownames(umap) <- umap[[1]]         # first column is cell ID
#umap <- as.matrix(umap[,-1])

weights <- read_csv("./bp candidate1/assemble Palantir/fate_bias.csv")
rownames(weights) <- weights [[1]]         # first column is cell ID
weights  <- as.matrix(weights[,-1])

# Assemble SCE
sce <- SingleCellExperiment(
  assays = list(counts = expr_mat),
  #reducedDims = SimpleList(UMAP = umap),
  colData = col_data
)

metadata(sce)$fate_bias <- weights
saveRDS(sce, file = "./bp candidate1/assemble Palantir/palantir_hvg_sce.rds")






library(Seurat)
library(tradeSeq)
library(RColorBrewer)
library(SingleCellExperiment)
library(slingshot)
library(zellkonverter)
library(Matrix)
library(readr)

setwd('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure4 Seacell Palantir/tradeseq new/bp candidate1/assemble Seacell')


# Read matrix (sparse or dense)
expr_mat <- Matrix::readMM("./expression_matrix.mtx")
expr_mat <- t(expr_mat)

# Read features and barcodes
genes <- read_csv("./features.csv", col_names = FALSE)[[1]]
genes = genes[2:length(genes)]
cells <- read_csv("./barcodes.csv", col_names = FALSE)[[1]]
cells = cells[2:length(cells)]

# Assign row and column names
rownames(expr_mat) <- genes
colnames(expr_mat) <- cells

# Read pseudotime
pseudotime <- read_csv("./pseudotime.csv")
rownames(pseudotime) <- cells  # Make sure rownames match colnames of matrix

col_data <- DataFrame(
  pseudotime = pseudotime[[2]]
  #leiden = factor(leiden[[1]])
)
rownames(col_data) <- cells


#umap <- read_csv("./assemble Palantir/X_umap.csv")
#rownames(umap) <- umap[[1]]         # first column is cell ID
#umap <- as.matrix(umap[,-1])

weights <- read_csv("./fate_bias.csv")
rownames(weights) <- weights [[1]]         # first column is cell ID
weights  <- as.matrix(weights[,-1])

# Assemble SCE
sce <- SingleCellExperiment(
  assays = list(counts = expr_mat),
  #reducedDims = SimpleList(UMAP = umap),
  colData = col_data
)

metadata(sce)$fate_bias <- weights
saveRDS(sce, file = "./seacell_allgenes_sce.rds")





