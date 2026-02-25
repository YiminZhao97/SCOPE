library(Iso)

# --- Set up paths ---
input_dir <- "/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure5 Epigenetic Priming/porportion analysis/res"
output_dir <- "/Users/zhaoyimin/Desktop/SCOPE Manuscipt/Figure5 Epigenetic Priming/porportion analysis/plots"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Get all CSV files
csv_files <- list.files(input_dir, pattern = "\\.csv$", full.names = TRUE)

# Process each CSV file
for (csv_file in csv_files) {

  # Extract TF name from filename
  base_name <- tools::file_path_sans_ext(basename(csv_file))

  # Read CSV
  df <- read.csv(csv_file)

  # Check if file has at least 3 columns (iteration, TF, enhancer columns)
  if (ncol(df) < 3) {
    cat(sprintf("Skipping %s: not enough columns\n", base_name))
    next
  }

  # Column 2 is TF importance
  tf_importance <- df[, 2]

  # Columns 3 onwards are enhancer importances - take average
  enhancer_importance <- rowMeans(df[, 3:ncol(df), drop = FALSE])

  # Get iteration values
  iteration <- df$iteration

  # Normalize by max value (0 to 1 scale)
  tf_norm <- tf_importance / max(tf_importance)
  enhancer_norm <- enhancer_importance / max(enhancer_importance)

  # Perform Unimodal Regression (ufit)
  fit_tf <- Iso::ufit(y = tf_norm, x = iteration, type = "raw")
  fit_enhancer <- Iso::ufit(y = enhancer_norm, x = iteration, type = "raw")

  # Create output filename
  output_file <- file.path(output_dir, paste0(base_name, "_priming_plot.pdf"))

  # Create plot
  pdf(output_file, width = 8, height = 6, family = "Helvetica")

  plot(iteration, enhancer_norm, pch = 16, col = rgb(0, 0, 1, 0.2),
       xlab = "Iteration",
       ylab = "Normalized Importance",
       main = paste("Epigenetic Priming:", base_name),
       xlim = c(max(iteration), min(iteration)),
       ylim = c(min(enhancer_norm, tf_norm), 1.1),
       axes = TRUE)

  # Add TF points
  points(iteration, tf_norm, pch = 16, col = rgb(1, 0, 0, 0.2))

  # Add fitted lines
  lines(iteration, fit_enhancer$y, col = "blue", lwd = 3)
  lines(iteration, fit_tf$y, col = "red", lwd = 3)

  # Add mode lines
  abline(v = fit_enhancer$mode, col = "blue", lty = 2, lwd = 2)
  abline(v = fit_tf$mode, col = "red", lty = 2, lwd = 2)

  # Add Legend
  legend("topleft",
         legend = c(paste("Enhancer (Peak:", length(enhancer_norm) - fit_enhancer$mode, ")"),
                    paste("TF (Peak:", length(tf_norm) - fit_tf$mode, ")")),
         col = c("blue", "red"), lwd = 3, lty = 1, bty = "n")

  dev.off()

  cat(sprintf("Created plot for %s\n", base_name))
}

cat("All plots completed!\n")
