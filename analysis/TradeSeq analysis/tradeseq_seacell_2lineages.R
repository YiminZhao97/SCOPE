library(Seurat)
library(tradeSeq)
library(RColorBrewer)
library(SingleCellExperiment)
library(slingshot)
library(zellkonverter)
library(Matrix)
library(readr)

setwd('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure4 Seacell Palantir/tradeseq new')

#only hvgs
sce = readRDS("./bp candidate1/assemble Seacell/seacell_allgenes_sce.rds")

#the psuedotime should be n by m where m is the number of lineage
pseudotime_matrix <- cbind(colData(sce)$pseudotime, 
                           colData(sce)$pseudotime)

weights <- metadata(sce)$fate_bias[,c(3,5)] / rowSums(metadata(sce)$fate_bias[,c(3,5)])
#we should avoid observation that rowSums(metadata(sce)$fate_bias[,c(2,3)]) == 0

valid_cells <- rowSums(metadata(sce)$fate_bias[, c(3,5)]) > 0
sce <- sce[, valid_cells]
pseudotime_matrix <- pseudotime_matrix[valid_cells, ]
weights <- weights[valid_cells, ]

sce <- fitGAM(counts = as.matrix(assay(sce, "counts")), pseudotime = pseudotime_matrix, 
              cellWeights = weights,
              nknots = 6, verbose = FALSE)


#we could imagine we need to subset the dataset
patternRes <- patternTest(sce)
oPat <- order(patternRes$waldStat, decreasing = TRUE)
head(rownames(patternRes)[oPat])

saveRDS(patternRes,  './bp candidate1/assemble Seacell/res_branchpoint_mono_ery_2lineages_allgenes.rds')
