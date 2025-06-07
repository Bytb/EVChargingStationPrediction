library(sf)
library(dplyr)
library(stringr)

# --- Load spatial data ---
segments <- st_read('Roads.shp')

# --- Get touching relationships ---
touches <- st_touches(segments, segments)

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
write.csv(edges_expanded, "edges_expanded.csv", row.names = FALSE)

cat("Expanded edge list written to edges_expanded.csv\n")
