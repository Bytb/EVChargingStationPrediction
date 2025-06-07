library(dplyr)
library(readr)
library(stringr)

# ----- CONFIG -----
main_feature_file <- "Raw_Features_Test.csv"
input_folder <- "AvgDistance_StationCount"
output_file <- "Final_Joined_Features.csv"

# ----- READ MAIN FEATURE FILE -----
main_df <- read_csv(main_feature_file, show_col_types = FALSE)

# ----- COMBINE ALL STATION COUNT FILES -----
station_files <- list.files(input_folder, pattern = "^StationCount_\\d{4}\\.csv$", full.names = TRUE)
station_df <- lapply(station_files, read_csv, show_col_types = FALSE) %>%
  bind_rows()

# ----- COMBINE ALL AVERAGE DISTANCE FILES -----
distance_files <- list.files(input_folder, pattern = "^AvgDistance_\\d{4}\\.csv$", full.names = TRUE)
distance_df <- lapply(distance_files, read_csv, show_col_types = FALSE) %>%
  bind_rows()

# ----- JOIN THE TWO FEATURE SETS -----
features_combined <- full_join(station_df, distance_df, by = "RoadYearID")

# ----- FINAL JOIN WITH MAIN FILE -----
final_df <- left_join(main_df, features_combined, by = "RoadYearID")

# ----- WRITE OUTPUT -----
write_csv(final_df, output_file)
message("✅ Final merged file saved as: ", output_file)
