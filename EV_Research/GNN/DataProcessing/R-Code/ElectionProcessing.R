# Load required libraries
library(dplyr)
library(readr)

# ---- 1. Read your election data ----
# Replace with your actual file name (CSV or TSV)
election_data <- read_csv("countypres_2000-2020.csv")

# ---- 2. Filter for 2016 and 2020 only ----
filtered_data <- election_data %>%
  filter(year %in% c(2016, 2020))

# ---- 3. For each county-year, keep the party with the most votes ----
winners <- filtered_data %>%
  group_by(county_fips, year) %>%
  slice_max(order_by = candidatevotes, n = 1, with_ties = FALSE) %>%
  ungroup()

# ---- 4. Add GEOID, rename candidatevotes to Votes, and keep only needed columns ----
result <- winners %>%
  mutate(
    GEOID = sprintf("%05d", county_fips),
    Party = party,
    Votes = candidatevotes
  ) %>%
  select(GEOID, Party, Votes, year)

# ---- 5. Split by year ----
result_2016 <- result %>%
  filter(year == 2016) %>%
  select(-year)  # Remove year column if you don't want it in output

result_2020 <- result %>%
  filter(year == 2020) %>%
  select(-year)

# ---- 6. Save to separate CSVs ----
write_csv(result_2016, "County_Winners_2016.csv")
write_csv(result_2020, "County_Winners_2020.csv")
