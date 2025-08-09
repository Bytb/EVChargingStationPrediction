library(sf)
library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(stringr)

# ---------- CONFIG ----------
years <- 2016:2023
year_suffixes <- 201:208
suffix_map <- setNames(as.character(year_suffixes), as.character(years))
features <- list(
  list(name = "Income", folder = "FeatureCSVs/Income_Cleaned", prefix = "Regist_"),
  list(name = "Population", folder = "FeatureCSVs/Population", prefix = "Regist_"),
  list(name = "White", folder = "FeatureCSVs/White_Cleaned", prefix = "Regist_"),
  list(name = "Age", folder = "FeatureCSVs/Age_Cleaned", prefix = "Regist_"),
  list(name = "Male", folder = "FeatureCSVs/Male_Cleaned", prefix = "Regist_"),
  list(name = "Education", folder = "FeatureCSVs/Education_Cleaned", prefix = "Regist_"),
  list(name = "SingleUnit", folder = "FeatureCSVs/SingleUnit_Cleaned", prefix = "Regist_")
)
tracts_folder <- "CensusTractPoints"
output_folder <- "ProcessedFeatures"

# Create output folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# ---------- FUNCTION ----------
process_feature <- function(feature_name, feature_folder, prefix) {
  all_years <- list()
  
  for (year in years) {
    suffix <- suffix_map[as.character(year)]
    
    # File paths
    tract_file <- file.path(tracts_folder, paste0("Tracts_", year, ".shp"))
    feature_file <- file.path(feature_folder, paste0(feature_name, "_", year, ".csv"))
    
    # Read tract shapefile and feature CSV
    tracts <- st_read(tract_file, quiet = TRUE)
    feat_df <- read_csv(feature_file, show_col_types = FALSE)
    
    # Read CSV and print actual column names
    feat_df <- read_csv(feature_file, show_col_types = FALSE)
    message("Columns in CSV: ", paste(names(feat_df), collapse = ", "))
    
    # Standardize column names safely
    colnames(feat_df)[1:2] <- c("GEOID", feature_name)
    
    # Ensure types match for joining
    feat_df$GEOID <- as.character(feat_df$GEOID)
    tracts$GEOID <- as.character(tracts$GEOID)
    
    # --- Drop geometry to do attribute join ---
    joined <- tracts %>%
      st_drop_geometry() %>%
      left_join(feat_df, by = "GEOID") %>%
      mutate(!!paste0(prefix, suffix) := replace_na(as.numeric(.data[[feature_name]]), 0)) %>%
      select(GEOID, paste0(prefix, suffix))
    
    all_years[[as.character(year)]] <- joined
  }
  
  # --- Combine all years ---
  combined <- reduce(all_years, full_join, by = "GEOID")
  
  # Fill remaining NAs (in case of tracts missing in all years)
  value_cols <- grep(prefix, names(combined), value = TRUE)
  combined[value_cols] <- lapply(combined[value_cols], \(x) replace_na(x, 0))
  
  # --- Join with latest geometry to restore spatial info ---
  base_tracts <- st_read(file.path(tracts_folder, "Tracts_2023.shp"), quiet = TRUE)
  final_geo <- left_join(base_tracts, combined, by = "GEOID")
  
  # Return the spatial feature table (sf object)
  return(final_geo)
}


# ---------- MAIN ----------
walk(features, ~ {
  result <- process_feature(.x$name, .x$folder, .x$prefix)
  
  # Output file paths
  shapefile_path <- file.path(output_folder, paste0(.x$name, "_Processed.shp"))
  csv_path <- file.path(output_folder, paste0(.x$name, "_Processed.csv"))
  
  # Write shapefile and CSV
  st_write(result, shapefile_path, delete_layer = TRUE)
  write_csv(st_drop_geometry(result), csv_path)
  
  message("✅ Saved outputs for feature: ", .x$name)
})

