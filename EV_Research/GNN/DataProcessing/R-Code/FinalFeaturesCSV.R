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
# station_files <- list.files(input_folder, pattern = "^StationCount_\\d{4}\\.csv$", full.names = TRUE)
# station_list <- lapply(station_files, read_csv, show_col_types = FALSE)
# 
# # Check if the combined file exists and include it
# # all_years_file <- file.path(input_folder, "StationCounts_AllYears.csv")
# # if (file.exists(all_years_file)) {
# #   message("📦 Including StationCount_AllYears.csv")
# #   all_years_df <- read_csv(all_years_file, show_col_types = FALSE)
# #   station_list <- append(station_list, list(all_years_df))
# # }
# 
# station_df <- bind_rows(station_list)
# ----- LOAD COMBINED STATION COUNT FILE ONLY -----
station_df <- read_csv(file.path(input_folder, "StationCounts_AllYears.csv"), show_col_types = FALSE)


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
