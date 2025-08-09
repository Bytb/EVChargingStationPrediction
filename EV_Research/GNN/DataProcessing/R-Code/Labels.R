# # ------------------- LIBRARIES -------------------
# library(sf)
# library(dplyr)
# library(stringr)
# library(tidyr)
# 
# # ------------------- PARAMETERS -------------------
# station_folder <- "Labels/FilteredStationLocationsPerYear"
# road_file <- "Segments/Roads.shp"
# buffer_miles <- 5
# year_range <- 2016:2025
# buffer_dist <- buffer_miles * 1609.34  # Convert miles to meters
# 
# # ------------------- LOAD ROADS -------------------
# cat("📦 Reading road shapefile...\n")
# roads <- st_read(road_file)
# 
# cat("📏 Reprojecting roads to UTM (EPSG:5070)...\n")
# roads <- st_transform(roads, 5070)
# 
# if (!"RoadID" %in% names(roads)) {
#   stop("❌ The road shapefile must have a field named 'RoadID'.")
# }
# 
# cat("🛞 Buffering roads by", buffer_miles, "miles...\n")
# roads_buffered <- st_buffer(roads, dist = buffer_dist)
# 
# # ------------------- PROCESS FUNCTION -------------------
# process_year_pair <- function(start_year) {
#   end_year <- start_year + 2
#   cat("🟡 Starting pair:", start_year, "-", end_year, "\n")
#   
#   if (end_year > max(year_range)) {
#     cat("⏭️ Skipping", start_year, "since", end_year, "is out of range.\n")
#     return(NULL)
#   }
#   
#   file_start <- file.path(station_folder, paste0("Stations_", start_year, ".shp"))
#   file_end   <- file.path(station_folder, paste0("Stations_", end_year, ".shp"))
#   
#   if (!file.exists(file_start) || !file.exists(file_end)) {
#     cat("❌ Missing shapefile for", start_year, "or", end_year, "- skipping\n")
#     return(NULL)
#   }
#   
#   cat("📂 Reading station files...\n")
#   stations_start <- st_read(file_start, quiet = FALSE) %>% st_transform(st_crs(roads))
#   stations_end   <- st_read(file_end, quiet = FALSE) %>% st_transform(st_crs(roads))
#   
#   cat("🔗 Performing spatial joins...\n")
#   count_start <- st_join(roads_buffered, stations_start, join = st_intersects) %>%
#     group_by(RoadID) %>%
#     summarise(StartCount = n(), .groups = "drop")
#   
#   count_end <- st_join(roads_buffered, stations_end, join = st_intersects) %>%
#     group_by(RoadID) %>%
#     summarise(EndCount = n(), .groups = "drop")
#   
#   count_start <- st_drop_geometry(count_start)
#   count_end   <- st_drop_geometry(count_end)
#   
#   result <- full_join(count_start, count_end, by = "RoadID") %>%
#     mutate(
#       StartCount = replace_na(StartCount, 0),
#       EndCount = replace_na(EndCount, 0),
#       Label = EndCount - StartCount,
#       RoadYear = paste0("Road", RoadID, "_", start_year)
#     ) %>%
#     select(RoadYear, Label)
#   
#   cat("✅ Finished pair:", start_year, "-", end_year, "\n")
#   return(result)
# }
# 
# # ------------------- MAIN LOOP -------------------
# results_list <- list()
# 
# for (year in year_range) {
#   tryCatch({
#     res <- process_year_pair(year)
#     if (!is.null(res)) results_list[[as.character(year)]] <- res
#   }, error = function(e) {
#     cat("❌ Error in year", year, ":", e$message, "\n")
#   })
# }
# 
# # ------------------- COMBINE AND SAVE -------------------
# cat("📊 Combining results...\n")
# results <- bind_rows(results_list)
# 
# cat("💾 Writing to FullScale_Labels.csv...\n")
# write.csv(results, "FullScale_Labels.csv", row.names = FALSE)
# 
# cat("🎉 Done! File saved as FullScale_Labels.csv\n")

library(sf)
library(dplyr)
library(stringr)
library(tidyr)


# PARAMETERS
#For testing, I set the working directory to ArcGISData -->
station_folder <- "Labels/FilteredStationLocationsPerYear"  # Folder with station shapefiles
road_file <- "Segments/Roads.shp"         # Road shapefile
buffer_miles <- 5
year_range <- 2016:2025
buffer_dist <- buffer_miles * 1609.34  # Convert miles to meters

# Load and reproject road segments
roads <- st_read(road_file)
roads <- st_transform(roads, 32617)  # Replace with your local projected CRS if needed

# Ensure road ID field is named "ID"
if (!"RoadID" %in% names(roads)) {
  stop("The road shapefile must have a field named 'ID' for RoadID.")
}

# Create 5-mile buffer
roads_buffered <- st_buffer(roads, dist = buffer_dist)

# Initialize results dataframe
results <- data.frame(RoadYear = character(), Label = integer(), stringsAsFactors = FALSE)

# Loop over years to compute 2-year label
for (start_year in year_range) {
  end_year <- start_year + 2
  if (end_year > max(year_range)) next
  
  # File paths
  file_start <- file.path(station_folder, paste0("Stations_", start_year, ".shp"))
  file_end <- file.path(station_folder, paste0("Stations_", end_year, ".shp"))
  
  # Skip if either file is missing
  if (!file.exists(file_start) || !file.exists(file_end)) {
    message(paste("Missing shapefile for", start_year, "or", end_year, "- skipping"))
    next
  }
  
  # Read and reproject station data
  stations_start <- st_read(file_start) %>% st_transform(st_crs(roads))
  stations_end <- st_read(file_end) %>% st_transform(st_crs(roads))
  
  # Count stations within buffer for both years
  count_start <- st_join(roads_buffered, stations_start, join = st_intersects) %>%
    group_by(RoadID) %>%
    summarise(StartCount = n(), .groups = "drop")
  
  count_end <- st_join(roads_buffered, stations_end, join = st_intersects) %>%
    group_by(RoadID) %>%
    summarise(EndCount = n(), .groups = "drop")
  
  # Drop geometry to safely do non-spatial join
  count_start <- st_drop_geometry(count_start)
  count_end <- st_drop_geometry(count_end)
  
  # Join, compute label, and format ID
  counts <- full_join(count_start, count_end, by = "RoadID") %>%
    mutate(
      StartCount = replace_na(StartCount, 0),
      EndCount = replace_na(EndCount, 0),
      Label = EndCount - StartCount,
      RoadYear = paste0("Road", RoadID, "_", start_year)
    ) %>%
    select(RoadYear, Label)
  
  # Append to results
  results <- bind_rows(results, counts)
}

# Save to CSV
write.csv(results, "Labels_Test.csv", row.names = FALSE)