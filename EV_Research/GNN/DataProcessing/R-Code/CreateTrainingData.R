library(sf)
library(dplyr)
library(purrr)
library(tidyr)
library(readr)

# --- CONFIG ---
road_file <- "Segments/Roads.shp"

vote_shapefile <- "PreppedFeatures/JoinedVotingData.shp"
temperature_file <- "PreppedFeatures/Temperature.shp"
feature_files <- list(
  list(file = "PreppedFeatures/Traffic_Prepped.shp", prefix = "Traffic"),
  list(file = "PreppedFeatures/RegisteredEV_Prepped.shp", prefix = "EVs"),
  list(file = "PreppedFeatures/Population_Processed.shp", prefix = "Population"),
  list(file = "PreppedFeatures/Income_Processed.shp", prefix = "Income"),
  list(file = "PreppedFeatures/Age_Processed.shp", prefix = "Age"),
  list(file = "PreppedFeatures/White_Processed.shp", prefix = "White"),
  list(file = "PreppedFeatures/Male_Processed.shp", prefix = "Male"),
  list(file = "PreppedFeatures/Education_Processed.shp", prefix = "Education"),
  list(file = "PreppedFeatures/SingleUnit_Processed.shp", prefix = "SingleUnit")
)
year_suffixes <- 201:208
year_cols <- paste0("Regist_", year_suffixes)
buffer_dist <- 8046.7  # 5 miles in meters
suffix_to_year <- setNames(as.character(2016:2023), as.character(201:208))

# --- FUNCTIONS ---
process_feature <- function(feature_file, prefix, road_buffers) {
  feat <- st_read(feature_file) %>% st_transform(st_crs(road_buffers)) %>%
    st_centroid()
  joined <- st_join(road_buffers, feat, join = st_intersects, left = TRUE)

  summary <- joined %>%
    st_drop_geometry() %>%
    group_by(RoadID) %>%
    summarise(across(all_of(year_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop") %>%
    rename_with(
      .fn = \(name) paste0(prefix, "_", sub("Regist_", "", name)),
      .cols = all_of(year_cols)
    )

  return(summary)
}

process_policy <- function(road_buffers, vote_sf) {
  joined <- st_join(road_buffers, vote_sf, join = st_intersects, left = FALSE)

  summary <- joined %>%
    st_drop_geometry() %>%
    group_by(RoadID) %>%
    summarise(
      Dem_2016 = sum(Dem_2016_V, na.rm = TRUE),
      Rep_2016 = sum(Rep_2016_V, na.rm = TRUE),
      Dem_2020 = sum(Dem_2020_V, na.rm = TRUE),
      Rep_2020 = sum(Rep_2020_V, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      Policy_201 = (Dem_2016 - Rep_2016) / (Dem_2016 + Rep_2016),
      Policy_202 = Policy_201,
      Policy_203 = Policy_201,
      Policy_204 = Policy_201,
      Policy_205 = (Dem_2020 - Rep_2020) / (Dem_2020 + Rep_2020),
      Policy_206 = Policy_205,
      Policy_207 = Policy_205,
      Policy_208 = Policy_205
    ) %>%
    select(RoadID, starts_with("Policy_"))

  return(summary)
}

process_temperature <- function(temperature_file, road_buffers) {
  temp_sf <- st_read(temperature_file) %>% st_transform(st_crs(road_buffers))

  joined <- st_join(road_buffers, temp_sf, join = st_intersects, left = FALSE)

  temp_cols <- paste0("AVG_", 2016:2023)

  summary <- joined %>%
    st_drop_geometry() %>%
    group_by(RoadID) %>%
    summarise(across(all_of(temp_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    rename_with(
      .fn = \(name) paste0("Temperature_", match(sub("AVG_", "", name), 2016:2023) + 200),
      .cols = all_of(temp_cols)
    )

  return(summary)
}

# --- MAIN PIPELINE ---

# 1. Read and buffer roads
roads <- st_read(road_file) %>%
  mutate(
    Length = as.numeric(st_length(geometry)) * 0.000621371  # meters to miles
  )
road_buffers <- st_buffer(roads, buffer_dist)

# 2. Process feature shapefiles
feature_summaries <- map(feature_files, ~ process_feature(.x$file, .x$prefix, road_buffers))

# 3. Add repeated Length by suffix
length_repeated <- roads %>%
  st_drop_geometry() %>%
  select(RoadID, Length) %>%
  crossing(Suffix = as.character(201:208)) %>%
  mutate(FeatureName = paste0("Length_", Suffix)) %>%
  select(RoadID, FeatureName, Length) %>%
  pivot_wider(names_from = FeatureName, values_from = Length)
feature_summaries <- append(feature_summaries, list(length_repeated))

# 4. Process Policy Feature
vote_sf <- st_read(vote_shapefile) %>% st_transform(st_crs(road_buffers))
policy_summary <- process_policy(road_buffers, vote_sf)
feature_summaries <- append(feature_summaries, list(policy_summary))
temperature_summary <- process_temperature(temperature_file, road_buffers)
feature_summaries <- append(feature_summaries, list(temperature_summary))

# 5. Combine everything
joined_features <- reduce(feature_summaries, full_join, by = "RoadID")

# 6. Pivot to long and reassemble RoadYearID
long_output <- joined_features %>%
  pivot_longer(
    cols = -RoadID,
    names_to = "Feature_Year",
    values_to = "Value"
  ) %>%
  separate(Feature_Year, into = c("Feature", "Suffix"), sep = "_") %>%
  mutate(
    Year = suffix_to_year[Suffix],
    RoadYearID = paste0("Road", RoadID, "_", Year)
  ) %>%
  select(RoadYearID, Feature, Value) %>%
  pivot_wider(names_from = Feature, values_from = Value) %>%
  arrange(RoadYearID)

# --- NA Check ---
na_rows <- long_output %>% filter(if_any(-1, is.na))
write_csv(long_output, "Raw_Features_Test.csv")
write_csv(na_rows, "DEBUG_Raw_Features_Test_NA.csv")

cat("✅ Saved Raw_Features_Test.csv\n")
cat("⚠️ NA diagnostics in DEBUG_Raw_Features_Test_NA.csv —", nrow(na_rows), "rows with NA\n")

# 7. Write result
if (file.exists("Raw_Features_Test.csv")) file.remove("Raw_Features_Test.csv")
write_csv(long_output, "Raw_Features_Test.csv")


# library(sf)
# library(dplyr)
# library(purrr)
# library(tidyr)
# library(readr)
# 
# # --- CONFIG ---
# road_file <- "Segments/Roads.shp"
# vote_shapefile <- "PreppedFeatures/JoinedVotingData.shp"
# temperature_file <- "PreppedFeatures/Temperature.shp"
# feature_files <- list(
#   list(file = "PreppedFeatures/Traffic_Prepped.shp", prefix = "Traffic"),
#   list(file = "PreppedFeatures/RegisteredEV_Prepped.shp", prefix = "EVs"),
#   list(file = "PreppedFeatures/Population_Processed.shp", prefix = "Population"),
#   list(file = "PreppedFeatures/Income_Processed.shp", prefix = "Income"),
#   list(file = "PreppedFeatures/Age_Processed.shp", prefix = "Age"),
#   list(file = "PreppedFeatures/White_Processed.shp", prefix = "White"),
#   list(file = "PreppedFeatures/Male_Processed.shp", prefix = "Male"),
#   list(file = "PreppedFeatures/Education_Processed.shp", prefix = "Education"),
#   list(file = "PreppedFeatures/SingleUnit_Processed.shp", prefix = "SingleUnit")
# )
# year_suffixes <- 201:208
# year_cols <- paste0("Regist_", year_suffixes)
# buffer_dist <- 8046.7  # 5 miles in meters
# suffix_to_year <- setNames(as.character(2016:2023), as.character(201:208))
# 
# # --- FUNCTIONS ---
# process_feature <- function(feature_file, prefix, road_buffers) {
#   feat <- st_read(feature_file, quiet = TRUE) %>% st_transform(st_crs(road_buffers)) %>% st_centroid()
#   joined <- st_join(road_buffers, feat, join = st_intersects, left = TRUE)
#   summary <- joined %>%
#     st_drop_geometry() %>%
#     group_by(RoadID) %>%
#     summarise(across(all_of(year_cols), ~ sum(.x, na.rm = TRUE)), .groups = "drop") %>%
#     rename_with(~ paste0(prefix, "_", sub("Regist_", "", .)), all_of(year_cols))
#   return(summary)
# }
# 
# process_policy <- function(road_buffers, vote_sf) {
#   joined <- st_join(road_buffers, vote_sf, join = st_intersects, left = FALSE)
#   summary <- joined %>%
#     st_drop_geometry() %>%
#     group_by(RoadID) %>%
#     summarise(
#       Dem_2016 = sum(Dem_2016_V, na.rm = TRUE),
#       Rep_2016 = sum(Rep_2016_V, na.rm = TRUE),
#       Dem_2020 = sum(Dem_2020_V, na.rm = TRUE),
#       Rep_2020 = sum(Rep_2020_V, na.rm = TRUE),
#       .groups = "drop"
#     ) %>%
#     mutate(
#       Policy_201 = (Dem_2016 - Rep_2016) / (Dem_2016 + Rep_2016),
#       Policy_202 = Policy_201,
#       Policy_203 = Policy_201,
#       Policy_204 = Policy_201,
#       Policy_205 = (Dem_2020 - Rep_2020) / (Dem_2020 + Rep_2020),
#       Policy_206 = Policy_205,
#       Policy_207 = Policy_205,
#       Policy_208 = Policy_205
#     ) %>%
#     select(RoadID, starts_with("Policy_"))
#   return(summary)
# }
# 
# process_temperature <- function(temperature_file, road_buffers) {
#   temp_sf <- st_read(temperature_file, quiet = TRUE) %>% st_transform(st_crs(road_buffers))
#   joined <- st_join(road_buffers, temp_sf, join = st_intersects, left = FALSE)
#   temp_cols <- paste0("AVG_", 2016:2023)
#   summary <- joined %>%
#     st_drop_geometry() %>%
#     group_by(RoadID) %>%
#     summarise(across(all_of(temp_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
#     rename_with(~ paste0("Temperature_", match(sub("AVG_", "", .), 2016:2023) + 200), all_of(temp_cols))
#   return(summary)
# }
# 
# # --- MAIN PIPELINE ---
# roads <- st_read(road_file, quiet = TRUE) %>%
#   st_transform(5070) %>%
#   mutate(Length = as.numeric(st_length(geometry)) * 0.000621371)
# road_buffers <- st_buffer(roads, buffer_dist)
# 
# feature_summaries <- map(feature_files, ~ process_feature(.x$file, .x$prefix, road_buffers))
# 
# length_repeated <- roads %>%
#   st_drop_geometry() %>%
#   select(RoadID, Length) %>%
#   crossing(Suffix = as.character(201:208)) %>%
#   mutate(FeatureName = paste0("Length_", Suffix)) %>%
#   select(RoadID, FeatureName, Length) %>%
#   pivot_wider(names_from = FeatureName, values_from = Length)
# feature_summaries <- append(feature_summaries, list(length_repeated))
# 
# vote_sf <- st_read(vote_shapefile, quiet = TRUE) %>% st_transform(st_crs(road_buffers))
# policy_summary <- process_policy(road_buffers, vote_sf)
# feature_summaries <- append(feature_summaries, list(policy_summary))
# 
# temperature_summary <- process_temperature(temperature_file, road_buffers)
# feature_summaries <- append(feature_summaries, list(temperature_summary))
# 
# joined_features <- reduce(feature_summaries, full_join, by = "RoadID")
# 
# long_output <- joined_features %>%
#   pivot_longer(cols = -RoadID, names_to = "Feature_Year", values_to = "Value") %>%
#   separate(Feature_Year, into = c("Feature", "Suffix"), sep = "_") %>%
#   mutate(Year = suffix_to_year[Suffix], RoadYearID = paste0("Road", RoadID, "_", Year)) %>%
#   select(RoadYearID, Feature, Value) %>%
#   pivot_wider(names_from = Feature, values_from = Value) %>%
#   arrange(RoadYearID)
# 
# # --- NA Check ---
# na_rows <- long_output %>% filter(if_any(-1, is.na))
# write_csv(long_output, "Raw_Features_Test.csv")
# write_csv(na_rows, "DEBUG_Raw_Features_Test_NA.csv")
# 
# cat("✅ Saved Raw_Features_Test.csv\n")
# cat("⚠️ NA diagnostics in DEBUG_Raw_Features_Test_NA.csv —", nrow(na_rows), "rows with NA\n")
