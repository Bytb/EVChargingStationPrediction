# Load libraries
library(readr)
library(dplyr)
library(tidyr)

# ---- 1. Read your original file ----
data <- read_csv("countypres_2000-2020.csv")

# Capitalize first letter of county_name, lowercase the rest
data <- data %>%
  mutate(
    county_name = paste0(
      toupper(substr(county_name, 1, 1)),
      tolower(substr(county_name, 2, nchar(county_name)))
    )
  )

# ---- 2. Filter for years 2016 and 2020 ----
filtered_data <- data %>%
  filter(year %in% c(2016, 2020))

# ---- 3. Keep only DEMOCRAT and REPUBLICAN ----
cleaned_data <- filtered_data %>%
  filter(party %in% c("DEMOCRAT", "REPUBLICAN"))

# ---- 4. Sum votes per county, year, party ----
summed_data <- cleaned_data %>%
  group_by(county_name, year, party) %>%
  summarise(total_votes = sum(candidatevotes, na.rm = TRUE), .groups = "drop")

# ---- 5. Pivot to wide format: DEMOCRATIC and REPUBLICAN columns ----
wide_data <- summed_data %>%
  mutate(
    PartyYear = paste0(
      ifelse(party == "DEMOCRAT", "Democratic", "Republican"),
      "_",
      year
    )
  ) %>%
  select(county_name, PartyYear, total_votes) %>%
  pivot_wider(
    names_from = PartyYear,
    values_from = total_votes
  )

# ---- 6. Split into two separate files ----
data_2016 <- wide_data %>%
  select(county_name, Democratic_2016, Republican_2016)

data_2020 <- wide_data %>%
  select(county_name, Democratic_2020, Republican_2020)

# ---- 7. Save to CSV ----
write_csv(data_2016, "Election2016_Raw.csv")
write_csv(data_2020, "Election2020_Raw.csv")

# ---- 8. Optional: check counts ----
cat("Counties in 2016:", nrow(data_2016), "\n")
cat("Counties in 2020:", nrow(data_2020), "\n")
