#version for all genes
library(Seurat)
library(tradeSeq)
library(RColorBrewer)
library(SingleCellExperiment)
library(slingshot)
library(zellkonverter)
library(Matrix)
library(readr)

setwd('/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure4 Seacell Palantir/tradeseq new')

sce = readRDS("./bp candidate1/assemble Palantir/palantir_allgenes_sce.rds")

#the psuedotime should be n by m where m is the number of lineage, since this is branchpoint of Ery and Mono, so m = 2
pseudotime_matrix <- cbind(colData(sce)$pseudotime, 
                           colData(sce)$pseudotime)

#normalize weights for 2 lineages
weights <- metadata(sce)$fate_bias[,c(1,3)] / rowSums(metadata(sce)$fate_bias[,c(1,3)])
sce <- fitGAM(counts = as.matrix(assay(sce, "counts")), pseudotime = pseudotime_matrix, 
              cellWeights = weights,
              nknots = 6, verbose = FALSE)

#we could imagine we need to subset the dataset
patternRes <- patternTest(sce)
oPat <- order(patternRes$waldStat, decreasing = TRUE)
head(rownames(patternRes)[oPat])

saveRDS(patternRes, './bp candidate1/assemble Palantir/res_allgenes_branchpoint_mono_ery_2lineages.rds')
