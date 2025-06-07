library(sf)
library(dplyr)
library(stringr)
library(units)
library(tidyr)

# ----- CONFIG -----
road_shapefile <- "Segments/Roads.shp"
station_folder <- "RawStationLocationsPerYear"
output_folder <- "AvgDistance_StationCount"

# Create output folder if it doesn't exist
if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# Use projected CRS in feet (EPSG:5070 - NAD83 / Conus Albers)
proj_crs <- 5070
buffer_miles <- 5
buffer_feet <- set_units(buffer_miles, "mi") %>% set_units("ft") %>% drop_units()

# ----- LOAD ROADS -----
roads <- st_read(road_shapefile, quiet = TRUE) %>%
  st_transform(crs = proj_crs) %>%
  mutate(RoadID = row_number())  # Ensures 1-to-1 join with Raw_Features_Test

# Get centroids for distance calculation
road_centroids <- st_centroid(roads)

# ----- PROCESS EACH YEAR -----
years <- 2016:2023
for (year in years) {
  message("🔄 Processing year: ", year)
  
  station_path <- file.path(station_folder, paste0("ev_", year, ".shp"))
  if (!file.exists(station_path)) {
    warning("❌ Station file not found: ", station_path)
    next
  }
  
  stations <- st_read(station_path, quiet = TRUE) %>%
    st_transform(crs = proj_crs)
  
  # ----- DEDUPLICATE STATIONS -----
  stations <- stations %>%
    mutate(geom_str = st_as_text(geometry)) %>%
    distinct(geom_str, .keep_all = TRUE) %>%
    select(-geom_str)
  
  # ----- DEDUPLICATE STATIONS BY COORDINATES (~10 ft) -----
  coords <- st_coordinates(stations)
  stations <- stations %>%
    mutate(
      x = round(coords[, 1], 1),
      y = round(coords[, 2], 1)
    ) %>%
    group_by(x, y) %>%
    slice(1) %>%
    ungroup() %>%
    select(-x, -y)
  
  # ----- BUFFER EACH ROAD SEGMENT -----
  roads_buffered <- st_buffer(roads, dist = buffer_feet)
  
  # ----- SPATIAL JOIN: STATIONS IN EACH BUFFER -----
  joined <- st_join(roads_buffered, stations, join = st_intersects)
  
  # ----- COUNT STATIONS PER ROAD SEGMENT -----
  count_df <- joined %>%
    st_drop_geometry() %>%
    group_by(RoadID) %>%
    summarise(StationCount = n(), .groups = "drop") %>%
    right_join(st_drop_geometry(roads %>% select(RoadID)), by = "RoadID") %>%
    mutate(
      StationCount = ifelse(is.na(StationCount), 0, StationCount),
      RoadYearID = paste0("Road", RoadID, "_", year)
    ) %>%
    select(RoadYearID, StationCount)
  
  
  # ----- AVERAGE DISTANCE (from centroid to stations within 5 miles) -----
  distance_df <- lapply(1:nrow(road_centroids), function(i) {
    road_pt <- road_centroids[i, ]
    dists <- st_distance(road_pt, stations)
    dists_mi <- set_units(dists, "ft") %>% set_units("mi") %>% drop_units()
    nearby <- which(dists_mi <= buffer_miles)
    avg_dist <- if (length(nearby) == 0) NA else mean(dists_mi[nearby])
    data.frame(RoadID = road_pt$RoadID, AvgDistance = avg_dist)
  }) %>%
    bind_rows() %>%
    mutate(
      RoadYearID = paste0("Road", RoadID, "_", year),
      AvgDistance = ifelse(is.na(AvgDistance), 5.1, AvgDistance)
    ) %>%
    select(RoadYearID, AvgDistance)
  
  # ----- SAVE -----
  write.csv(count_df, file.path(output_folder, paste0("StationCount_", year, ".csv")), row.names = FALSE)
  write.csv(distance_df, file.path(output_folder, paste0("AvgDistance_", year, ".csv")), row.names = FALSE)
}
