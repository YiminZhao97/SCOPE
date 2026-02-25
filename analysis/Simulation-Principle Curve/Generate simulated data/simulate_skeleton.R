coords <- matrix(c(-20, 10, 25, 100, 
                   60, 80, 25, 100, 
                   40, 10, 60, 80, 
                   60, 80, 100, 25), 
                 nrow = 4, byrow = TRUE)

jpeg("./sim2/plot_lines—test.jpeg", width = 1600, height = 1400, res = 300)
# Set up the plot
plot(NULL, xlim = c(-20, 100), ylim = c(0, 100), xlab = "X", ylab = "Y", main = "Plot of Lines Based on Coordinates")
# Add the lines using the coordinates
segments(x0 = coords[1, 1], y0 = coords[1, 2], x1 = coords[1, 3], y1 = coords[1, 4], col = "blue", lwd = 2)  # First line
segments(x0 = coords[2, 1], y0 = coords[2, 2], x1 = coords[2, 3], y1 = coords[2, 4], col = "red", lwd = 2)   # Second line
segments(x0 = coords[3, 1], y0 = coords[3, 2], x1 = coords[3, 3], y1 = coords[3, 4], col = "green", lwd = 2) # Third line
segments(x0 = coords[4, 1], y0 = coords[4, 2], x1 = coords[4, 3], y1 = coords[4, 4], col = "purple", lwd = 2) # Fourth line
dev.off()


rm(list=ls())
library(ggplot2)
source("/Users/zhaoyimin/Desktop/Kevin/branch-point-prediction/Simulation/principle curve/generator.R")

para = list(c(200, 200, 200, 200), 120, 5,
            2, 50, 1/250, 1000,
            50)

names(para) = c("n_each", "d_each", "sigma",
                "k", "max_iter", "modifier", "max_val",
                "size")

# let's use the convention that the first point of each line segment is the "end", and the second point is the "start"
cell_pop <- matrix(c(-20, 10, 25, 100, 
                     60, 80, 25, 100, 
                     40, 10, 60, 80, 
                     60, 80, 100, 25), 
                   nrow = 4, byrow = TRUE)

#coordinate of start points and end points
gene_pop <- matrix(c(20,90, 25,100,
                     90,20, 100,25)/20, nrow = 2, ncol = 4, byrow = T)

n_each <- para[["n_each"]]
d_each <- para[["d_each"]]
sigma <- para[["sigma"]]
modifier <- para[["modifier"]]

set.seed(10)
res <- generate_natural_mat(cell_pop, gene_pop, n_each, d_each, sigma, modifier)
nat_mat <- res$nat_mat

mean_vec <- Matrix::colMeans(nat_mat)
sd_vec <- apply(nat_mat, 1, stats::sd)
for(j in 1:ncol(nat_mat)){
  nat_mat[,j] <- (nat_mat[,j] - mean_vec[j])/sd_vec[j]
}
pca_res <- stats::prcomp(nat_mat, center = FALSE, scale. = FALSE)
plot(pca_res$x[,1:2])


#derive pseudotime
pos = res$pos 
pos1 = pos[1:200]
pos2 = pos[201:400]
pos3 = pos[401:600]
pos4 = pos[601:800]
#segment1 
pseudotime = c(pos1)
#segment2
pseudotime = c(pseudotime, pos2 * 0.5)
#segment3
pseudotime = c(pseudotime, pos3 * 0.5 + 0.5)
#segment4
pseudotime = c(pseudotime, 0.5 - pos4 * 0.5+ 0.5)

dat = data.frame(nat_mat)
colnames(dat) = c(paste0("gene", 1:240))
rownames(dat) = c(paste0("cell", 1:800))

#2d plot colored by pseudotime
dat_pca = data.frame('x' = pca_res$x[,1], 'y' = pca_res$x[,2],
                     'pseudotime' = pseudotime)

# Check pesutotime
ggplot(dat_pca, aes(x = x, y = y, color = pseudotime)) +
  geom_point(size = 3) +  # size can be adjusted based on your preference
  scale_color_viridis_c() +  # Using a viridis color scale for pseudotime
  labs(x = 'PC1', y = 'PC2', color = 'Pseudotime', title = 'PCA embedding, color by pseudotime') +  # Add axis labels and color legend
  theme_minimal()  # Use a clean, minimal theme

ggsave("./sim2/pca_pseudotime_plot_sim2.png", width = 8, height = 6, dpi = 300)
pseudotime = data.frame('pseudotime' = pseudotime)
write.csv(pseudotime, file = "./sim2/principle_curve_pseudotime_sim2.csv", row.names = FALSE)
write.csv(dat, file = "./sim2/dat_sim2.csv", row.names = TRUE)