library(ggplot2)
library(dplyr)
library(sf)
library(viridis)
library(ggspatial)
library(akima)
library(rnaturalearth)  # For higher quality country boundaries
library(rnaturalearthdata)

# Temperature data
temp_data <- data.frame(
  station = c("Dhaka", "Chattogram", "Sylhet", "Rajshahi", "Rangpur",
              "Mymensingh", "Barishal", "Khulna", "Jessore", "Bogra",
              "Cumilla", "Srimangal", "Cox's Bazar", "Faridpur", "Rangamati", "Bhola"),
  lat = c(23.81, 22.36, 24.89, 24.37, 25.75, 24.75, 22.70, 22.84, 23.17, 24.85,
          23.46, 24.30, 21.45, 23.61, 23.14, 22.68),
  lon = c(90.41, 91.78, 91.86, 88.60, 89.25, 90.42, 90.35, 89.54, 89.21, 89.31,
          91.18, 91.72, 92.02, 89.84, 92.14, 90.65),
  max_temp = c(34.77, 33.14, 32.83, 36.30, 32.12, 32.39, 34.14, 35.38, 36.05, 33.72,
               33.20, 33.52, 33.54, 35.28, 34.77, 33.28)
)

# Get high-resolution Bangladesh boundary
bd_shape <- ne_states(country = "Bangladesh", returnclass = "sf")
bd_boundary <- st_union(bd_shape) # Unified country boundary

# Create interpolation grid (increase grid resolution)
grid <- expand.grid(
  lon = seq(88, 92.5, length.out = 500),  # Increased resolution
  lat = seq(20.5, 26.5, length.out = 500)  # Increased resolution
)

# Perform interpolation with extrapolation
interp_data <- interp(
  x = temp_data$lon,
  y = temp_data$lat,
  z = temp_data$max_temp,
  xo = seq(88, 92.5, length = 500),  # Increased grid resolution
  yo = seq(20.5, 26.5, length = 500),
  linear = FALSE,  # Cubic interpolation for smoother results
  extrap = TRUE  # Allow extrapolation to cover the full map area
)

# Create data frame for plotting
interp_df <- data.frame(
  lon = rep(interp_data$x, each = length(interp_data$y)),
  lat = rep(interp_data$y, length(interp_data$x)),
  temp = as.vector(interp_data$z)
)

# Clip to Bangladesh boundary
grid_sf <- st_as_sf(interp_df, coords = c("lon", "lat"), crs = 4326)
grid_inside <- st_intersection(grid_sf, st_make_valid(bd_boundary))
interp_df <- as.data.frame(st_coordinates(grid_inside))
interp_df$temp <- grid_inside$temp

# Create the plot
ggplot() +
  # Filled temperature contours with subdivisions
  geom_tile(data = interp_df, aes(x = X, y = Y, fill = temp)) +
  
  # Administrative subdivisions
  geom_sf(data = bd_shape, fill = NA, color = "gray40", size = 0.3, alpha = 0.5) +
  
  # Country boundary
  geom_sf(data = bd_boundary, fill = NA, color = "black", size = 0.8) +
  
  # Station points
  geom_point(data = temp_data, aes(x = lon, y = lat), 
             color = "black", size = 3, shape = 21, fill = "white") +
  
  # Station labels with better positioning
  ggrepel::geom_text_repel(
    data = temp_data,
    aes(x = lon, y = lat, label = paste0(station, "\n", max_temp, "°C")),
    size = 3, color = "black", fontface = "bold",
    box.padding = 0.5, min.segment.length = 0
  ) +
  
  # Color gradient with precise legend
  scale_fill_viridis(
    name = "Temperature (°C)",
    option = "plasma",
    limits = c(32, 36.5),
    breaks = seq(32, 36.5, by = 0.5),
    guide = guide_colorbar(
      barwidth = 1, barheight = 15,
      title.position = "right",
      title.theme = element_text(angle = 90, hjust = 0.5),
      frame.colour = "black",
      ticks.colour = "black"
    )
  ) +
  
  # Contour lines
  geom_contour(
    data = interp_df, aes(x = X, y = Y, z = temp),
    color = "white", alpha = 0.5, size = 0.3,
    breaks = seq(32, 36.5, by = 0.5)
  ) +
  
  # Map styling
  coord_sf(xlim = c(88, 92.5), ylim = c(20.5, 26.5)) +
  labs(
    title = "Maximum Temperature Distribution in Bangladesh",
    subtitle = "Interpolated from weather station data with administrative boundaries",
    caption = "Data source: Meteorological stations"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    panel.background = element_rect(fill = "lightblue"),
    panel.grid = element_line(color = "gray90", size = 0.2)
  )
