# Load required package
library(readr)

# Set input and output folder paths
input_folder <- "AvgWhiteMaleAge"
output_folder <- "AvgWhiteMaleAge_Split"

# Create output folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# Define the years to process
years <- 2016:2023

# Loop through each year's CSV file
for (year in years) {
  # Get the two-digit suffix (e.g., 2016 -> 16)
  yy <- sprintf("%02d", year %% 100)
  
  # Construct the input file path
  file_path <- file.path(input_folder, paste0(year, ".csv"))
  
  # Read the data
  df <- read_csv(file_path, show_col_types = FALSE)
  
  # Rename GEO_ID to GEOID
  colnames(df)[1] <- "GEOID"
  
  # Define feature prefixes and capitalized output names
  features <- c("Male", "age", "white")
  output_names <- c("Male", "Age", "White")
  
  for (i in seq_along(features)) {
    input_col <- paste0(features[i], yy)         # e.g., Male17
    output_col <- paste0(output_names[i], "_", year)  # e.g., Male_2017
    output_file <- file.path(output_folder, paste0(output_names[i], "_", year, ".csv"))
    
    # Extract relevant columns and rename
    temp_df <- df[, c("GEOID", input_col)]
    colnames(temp_df)[2] <- output_col
    
    # Write to CSV
    write_csv(temp_df, output_file)
  }
}
