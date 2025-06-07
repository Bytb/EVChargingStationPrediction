library(sf)
library(dplyr)
library(tidyr)
library(purrr)
library(readr)

# ---------- CONFIG ----------
road_file <- "Segments/Roads.shp"
feature_files <- list(
  list(file = "RawFeatures/RawTraffic.shp", prefix = "Traffic"),
  list(file = "RawFeatures/RawRegisteredEVs.shp", prefix = "EVs"),
  list(file = "RawFeatures/Population_Processed.shp", prefix = "Population"),
  list(file = "RawFeatures/Income_Processed.shp", prefix = "Income"),
  list(file = "RawFeatures/Age_Processed.shp", prefix = "Age"),
  list(file = "RawFeatures/White_Processed.shp", prefix = "White"),
  list(file = "RawFeatures/Male_Processed.shp", prefix = "Male"),
  list(file = "RawFeatures/Education_Processed.shp", prefix = "Education"),
  list(file = "RawFeatures/SingleUnit_Processed.shp", prefix = "SingleUnit")
)
year_suffixes <- 201:208
year_cols <- paste0("Regist_", year_suffixes)
buffer_dist <- 8046.7

# ---------- FUNCTION ----------
process_feature <- function(feature_file, prefix, road_buffers) {
  feat <- st_read(feature_file) %>% st_transform(st_crs(road_buffers))
  joined <- st_join(road_buffers, feat, join = st_contains, left = FALSE)
  
  summary <- joined %>%
    st_drop_geometry() %>%
    group_by(RoadID) %>%
    summarise(across(all_of(year_cols), \(x) sum(x, na.rm = TRUE)), .groups = "drop") %>%
    rename_with(
      .fn = \(name) paste0(prefix, "_", sub("Regist_", "", name)),
      .cols = all_of(year_cols)
    )
  
  return(summary)
}

# ---------- MAIN ----------

# Read and buffer road segments
roads <- st_read(road_file) %>% mutate(RoadID = row_number())
road_buffers <- st_buffer(roads, buffer_dist)

# Process all feature classes
feature_summaries <- map(feature_files, ~ process_feature(.x$file, .x$prefix, road_buffers))

# Define mapping from suffix to actual year
suffix_to_year <- setNames(as.character(2016:2023), as.character(201:208))

# Combine all feature summaries by RoadID
joined_features <- reduce(feature_summaries, full_join, by = "RoadID")

# Pivot to long format and fix year labels
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
  pivot_wider(
    names_from = Feature,
    values_from = Value
  ) %>%
  arrange(RoadYearID)


# Export
if (file.exists("Road_Feature_By_Year.csv")) {
  file.remove("Road_Feature_By_Year.csv")
}
write_csv(long_output, "Road_Feature_By_Year.csv")

