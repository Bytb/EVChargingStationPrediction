library(sf)
library(dplyr)
library(stringr)

# --- Load spatial data ---
segments <- st_read('Segments/Roads.shp')

# --- Get touching relationships ---
segments <- st_transform(segments, 5070)
touches <- st_touches(segments, segments)

isolated_count <- sum(lengths(touches) == 0)
cat("Isolated segments:", isolated_count, "\n")


# --- Create base edge list from touches ---
edge_list <- do.call(rbind, lapply(1:length(touches), function(i) {
  data.frame(source = i, target = touches[[i]])
}))

# --- Remove duplicates if the network is undirected ---
edge_list <- edge_list %>% distinct()

# --- Define year range ---
years <- 2016:2023

# --- Expand edge list for each year ---
expanded_edges <- lapply(years, function(y) {
  edge_list %>%
    mutate(
      source = paste0("Road", source, "_", y),
      target = paste0("Road", target, "_", y)
    )
})

# --- Combine and write final edge list ---
edges_expanded <- bind_rows(expanded_edges)

# --- Create temporal edges ---
temporal_edges <- do.call(rbind, lapply(1:(length(years)-1), function(i) {
  year1 <- years[i]
  year2 <- years[i+1]
  
  data.frame(
    source = paste0("Road", 1:nrow(segments), "_", year1),
    target = paste0("Road", 1:nrow(segments), "_", year2)
  )
}))

# Optionally add reverse direction (if undirected)
temporal_edges_rev <- temporal_edges %>%
  rename(source = target, target = source)

# Combine spatial + temporal
edges_full <- bind_rows(edges_expanded, temporal_edges, temporal_edges_rev) %>%
  distinct()

# --- Save updated edge list ---
write.csv(edges_full, "edges_fullscale_with_time.csv", row.names = FALSE)

cat("✅ Full spatiotemporal edge list written to edges_fullscale_with_time.csv\n")






