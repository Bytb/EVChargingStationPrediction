library(readr)
library(dplyr)
library(stringr)

# ----- CONFIG -----
input_dir <- "SingleUnit"               # Change this as needed
output_dir <- "SingleUnit_Cleaned"      # Use input_dir to overwrite originals if desired

# Create output directory if needed
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# List all CSV files matching the pattern: Feature_YYYY.csv
csv_files <- list.files(input_dir, pattern = "^\\w+_\\d{4}\\.csv$", full.names = TRUE)

# Process each file
for (file_path in csv_files) {
  message("📂 Processing: ", file_path)
  
  # Extract feature name and year from filename
  filename <- basename(file_path)
  parts <- str_match(filename, "^(\\w+)_([0-9]{4})\\.csv$")
  feature <- parts[2]       # e.g., "White", "Income", "Male"
  year <- parts[3]          # e.g., "2016"
  
  # Read file
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # Rename first column to GEOID and extract last 11 characters
  colnames(df)[1] <- "GEOID"
  df$GEOID <- str_sub(as.character(df$GEOID), -11, -1)
  
  # Save cleaned file
  output_path <- file.path(output_dir, filename)
  write_csv(df, output_path)
  message("✅ Saved cleaned file to: ", output_path)
}
