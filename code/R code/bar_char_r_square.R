library(ggplot2)
library(dplyr)

# Corrected data with all updated MAE values
metrics_data <- data.frame(
  Station = rep(c("Dhaka", "Chittagong", "Barisal", "Bogura", "Comilla", 
                  "Cox's Bazar", "Faridpur", "Jessore", "Khulna", "Rajshahi",
                  "Rangamati", "Rangpur", "Sylhet", "Srimongol", "Bhola", 
                  "Mymensingh"), each = 9),
  Model = rep(c("ARIMA", "LSTM", "XGBoost", "ANN", "SVR", "SARIMA+LSTM", 
                "SARIMA+ANN", "SARIMA+XGBoost", "SARIMA+SVR"), 16),
  MAE = c(
    # Dhaka
    0.9616, 0.8071, 0.8818, 0.7906, 0.8210, 0.8320, 0.8510, 0.8323, 0.7457,
    # Chittagong
    0.8814, 0.8594, 0.7431, 0.6360, 0.8109, 0.9816, 1.0012, 0.9433, 0.9044,
    # Barisal
    0.7540, 0.6510, 0.7957, 0.6812, 0.6757, 0.7168, 0.7106, 0.6794, 0.6633,
    # Bogura
    0.7817, 0.9881, 0.8328, 1.0036, 0.8411, 0.7914, 0.7823, 0.8076, 0.7848,
    # Comilla
    0.9295, 0.7599, 0.8328, 0.7456, 0.7470, 0.7187, 0.7217, 0.6988, 0.7964,
    # Cox's Bazar
    0.7729, 0.6668, 0.7180, 0.6757, 0.8637, 1.4846, 1.4868, 1.5957, 0.7864,
    # Faridpur
    0.7675, 0.9398, 0.9143, 0.8027, 0.8119, 0.7690, 0.7657, 0.7412, 0.7762,
    # Jessore
    0.7036, 1.0839, 0.7384, 0.8037, 0.7584, 0.7105, 0.7081, 0.7401, 0.7523,
    # Khulna
    0.7324, 0.9081, 0.8292, 0.7850, 0.7604, 0.7366, 0.7444, 0.6820, 0.7481,
    # Rajshahi
    0.9251, 1.3846, 0.9633, 0.9883, 1.0441, 0.9626, 0.9690, 0.9027, 0.9692,
    # Rangamati
    1.0290, 0.8795, 0.8795, 0.9544, 0.8723, 0.8092, 0.8017, 0.8077, 0.7916,
    # Rangpur
    1.1648, 0.9926, 1.1819, 1.0435, 1.0595, 0.9841, 0.9865, 0.9574, 0.9949,
    # Sylhet
    0.9948, 0.9310, 1.0751, 0.9465, 0.9913, 0.8504, 0.8539, 0.8482, 0.8740,
    # Srimongol
    1.0672, 0.7972, 1.0034, 0.9025, 0.8712, 0.9299, 0.9304, 0.9033, 0.9842,
    # Bhola
    0.9102, 0.7796, 1.0354, 0.8004, 0.9222, 0.9436, 0.9329, 0.8641, 0.9491,
    # Mymensingh
    0.9254, 0.8628, 0.8732, 0.8247, 0.8396, 0.8087, 0.8056, 0.7499, 0.9135
  )
)

# Color palette for models
model_colors <- c(
  "ARIMA" = "#1f77b4", "LSTM" = "#ff7f0e", "XGBoost" = "#2ca02c",
  "ANN" = "#d62728", "SVR" = "#9467bd", "SARIMA+LSTM" = "#8c564b",
  "SARIMA+ANN" = "#e377c2", "SARIMA+XGBoost" = "#7f7f7f", "SARIMA+SVR" = "#bcbd22"
)

# Plot
ggplot(metrics_data, aes(x = Station, y = MAE, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "MAE Comparison Across 16 Stations",
    subtitle = "Performance of 9 Forecasting Models",
    x = "Station",
    y = "Mean Absolute Error (MAE)",
   
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.major.x = element_blank()
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))

#########################################################################################
library(ggplot2)
library(dplyr)

# Create dataframe with R² values
r2_data <- data.frame(
  Station = rep(c("Dhaka", "Chittagong", "Barisal", "Bogura", "Comilla", 
                  "Cox's Bazar", "Faridpur", "Jessore", "Khulna", "Rajshahi",
                  "Rangamati", "Rangpur", "Sylhet", "Srimongol", "Bhola", 
                  "Mymensingh"), each = 9),
  Model = rep(c("ARIMA", "LSTM", "XGBoost", "ANN", "SVR", "SARIMA+LSTM", 
                "SARIMA+ANN", "SARIMA+XGBoost", "SARIMA+SVR"), 16),
  R2 = c(
    # Dhaka
    0.8569, 0.8934, 0.8977, 0.8953, 0.9014, 0.9141, 0.9139, 0.8910, 0.9067,
    # Chittagong
    0.7707, 0.7774, 0.8751, 0.8828, 0.8635, 0.8760, 0.8759, 0.7721, 0.7881,
    # Barisal
    0.8857, 0.9171, 0.9065, 0.9084, 0.9189, 0.9280, 0.9264, 0.9057, 0.9102,
    # Bogura
    0.9035, 0.8490, 0.8911, 0.8741, 0.8937, 0.9073, 0.9078, 0.8989, 0.9007,
    # Comilla
    0.8219, 0.8733, 0.8911, 0.8887, 0.8833, 0.8927, 0.8927, 0.8882, 0.8509,
    # Cox's Bazar
    0.7427, 0.8184, 0.7968, 0.8224, 0.8953, 0.7453, 0.7449, 0.6002, 0.9004,
    # Faridpur
    0.9102, 0.8794, 0.9099, 0.9144, 0.9181, 0.9268, 0.9267, 0.9219, 0.9151,
    # Jessore
    0.9340, 0.8210, 0.9291, 0.9195, 0.9273, 0.9352, 0.9358, 0.9209, 0.9189,
    # Khulna
    0.9279, 0.8870, 0.9157, 0.9177, 0.9298, 0.9374, 0.9376, 0.9340, 0.9242,
    # Rajshahi
    0.9135, 0.7974, 0.9273, 0.9218, 0.9205, 0.9278, 0.9279, 0.9222, 0.9161,
    # Rangamati
    0.7991, 0.8469, 0.8469, 0.8359, 0.8455, 0.8560, 0.8571, 0.8499, 0.8539,
    # Rangpur
    0.7963, 0.8644, 0.8621, 0.8658, 0.8740, 0.8947, 0.8966, 0.8744, 0.8644,
    # Sylhet
    0.7637, 0.7958, 0.7880, 0.7752, 0.7604, 0.8146, 0.8147, 0.8115, 0.8013,
    # Srimongol
    0.7805, 0.8762, 0.8315, 0.8485, 0.8750, 0.8552, 0.8556, 0.8257, 0.8004,
    # Bhola
    0.8382, 0.8774, 0.8858, 0.8760, 0.8867, 0.8866, 0.8857, 0.8385, 0.8064,
    # Mymensingh
    0.8500, 0.8674, 0.8795, 0.8700, 0.8808, 0.9015, 0.9023, 0.8947, 0.8545
  )
)

# Use the same color palette as MAE plot
model_colors <- c(
  "ARIMA" = "#1f77b4", "LSTM" = "#ff7f0e", "XGBoost" = "#2ca02c",
  "ANN" = "#d62728", "SVR" = "#9467bd", "SARIMA+LSTM" = "#8c564b",
  "SARIMA+ANN" = "#e377c2", "SARIMA+XGBoost" = "#7f7f7f", "SARIMA+SVR" = "#bcbd22"
)

# Create R² plot
ggplot(r2_data, aes(x = Station, y = R2, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = expression(paste(R^2, " Comparison Across 16 Stations")),
    subtitle = "Performance of 9 Forecasting Models",
    x = "Station",
    y = expression(R^2),
    caption = "Data: Your Research"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.major.x = element_blank()
  ) +
  scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0, 0.1)))

##################################################################

library(ggplot2)
library(dplyr)

# Create dataframe with RMSE values
rmse_data <- data.frame(
  Station = rep(c("Dhaka", "Chittagong", "Barisal", "Bogura", "Comilla", 
                  "Cox's Bazar", "Faridpur", "Jessore", "Khulna", "Rajshahi",
                  "Rangamati", "Rangpur", "Sylhet", "Srimongol", "Bhola", 
                  "Mymensingh"), each = 9),
  Model = rep(c("ARIMA", "LSTM", "XGBoost", "ANN", "SVR", "SARIMA+LSTM", 
                "SARIMA+ANN", "SARIMA+XGBoost", "SARIMA+SVR"), 16),
  RMSE = c(
    # Dhaka
    1.1749, 1.0201, 1.1203, 1.0217, 1.0247, 1.0258, 1.0456, 1.0221, 0.9458,
    # Chittagong
    1.0524, 1.0672, 0.9087, 0.7955, 0.9634, 1.1349, 1.1566, 1.0957, 1.0566,
    # Barisal
    0.8927, 0.7836, 1.0107, 0.8931, 0.8347, 0.8721, 0.8675, 0.8363, 0.8163,
    # Bogura
    1.0005, 1.2495, 1.1011, 1.2295, 1.0560, 1.0080, 1.0010, 1.0240, 1.0151,
    # Comilla
    1.0906, 0.9525, 1.1011, 0.9345, 0.9435, 0.9160, 0.9183, 0.8933, 1.0316,
    # Cox's Bazar
    0.9684, 0.8041, 0.8801, 0.8355, 1.08, 1.8547, 1.8542, 2.0369, 1.0165,
    # Faridpur
    0.9662, 1.1820, 1.1200, 0.9987, 0.9928, 0.9180, 0.9149, 0.9052, 0.9440,
    # Jessore
    0.8696, 1.4117, 0.9346, 1.0360, 0.9356, 0.9108, 0.9054, 0.9524, 0.9640,
    # Khulna
    0.8822, 1.1042, 1.0348, 0.9886, 0.9349, 0.8874, 0.8945, 0.8441, 0.9048,
    # Rajshahi
    1.1538, 1.8206, 1.1530, 1.1878, 1.2549, 1.1382, 1.1430, 1.0991, 1.1418,
    # Rangamati
    1.1922, 1.1146, 1.1146, 1.1698, 1.0844, 1.0218, 1.0142, 1.0153, 1.0016,
    # Rangpur
    1.4132, 1.2913, 1.4575, 1.3605, 1.2953, 1.2073, 1.2136, 1.1797, 1.2255,
    # Sylhet
    1.2155, 1.1429, 1.3151, 1.2140, 1.2291, 1.0716, 1.0767, 1.0782, 1.1069,
    # Srimongol
    1.2802, 0.9947, 1.2714, 1.1492, 1.0795, 1.1870, 1.1872, 1.1593, 1.2405,
    # Bhola
    1.0543, 0.9559, 1.2643, 1.0040, 1.1237, 1.1514, 1.1379, 1.0603, 1.1609,
    # Mymensingh
    1.1305, 1.0794, 1.0835, 1.1614, 1.0545, 1.0066, 1.0011, 0.9494, 1.1162
  )
)

# Use the same color palette as previous plots
model_colors <- c(
  "ARIMA" = "#1f77b4", "LSTM" = "#ff7f0e", "XGBoost" = "#2ca02c",
  "ANN" = "#d62728", "SVR" = "#9467bd", "SARIMA+LSTM" = "#8c564b",
  "SARIMA+ANN" = "#e377c2", "SARIMA+XGBoost" = "#7f7f7f", "SARIMA+SVR" = "#bcbd22"
)

# Create RMSE plot
ggplot(rmse_data, aes(x = Station, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "RMSE Comparison Across 16 Stations",
    subtitle = "Performance of 9 Forecasting Models",
    x = "Station",
    y = "Root Mean Squared Error (RMSE)",
    caption = "Data: Your Research"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    panel.grid.major.x = element_blank()
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))




####################################

library(ggplot2)
library(dplyr)
library(tidyr)

# Prepare combined MAE and RMSE data
metrics_data <- data.frame(
  Station = rep(c("Dhaka", "Chittagong", "Barisal", "Bogura", "Comilla", 
                  "Cox's Bazar", "Faridpur", "Jessore", "Khulna", "Rajshahi",
                  "Rangamati", "Rangpur", "Sylhet", "Srimongol", "Bhola", 
                  "Mymensingh"), each = 9),
  Model = rep(c("ARIMA", "LSTM", "XGBoost", "ANN", "SVR", "SARIMA+LSTM", 
                "SARIMA+ANN", "SARIMA+XGBoost", "SARIMA+SVR"), 16),
  MAE = c(
    0.9616, 0.8071, 0.8818, 0.7906, 0.8210, 0.8320, 0.8510, 0.8323, 0.7457,
    0.8814, 0.8594, 0.7431, 0.6360, 0.8109, 0.9816, 1.0012, 0.9433, 0.9044,
    0.7540, 0.6510, 0.7957, 0.6812, 0.6757, 0.7168, 0.7106, 0.6794, 0.6633,
    0.7817, 0.9881, 0.8328, 1.0036, 0.8411, 0.7914, 0.7823, 0.8076, 0.7848,
    0.9295, 0.7599, 0.8328, 0.7456, 0.7470, 0.7187, 0.7217, 0.6988, 0.7964,
    0.7729, 0.6668, 0.7180, 0.6757, 0.8637, 1.4846, 1.4868, 1.5957, 0.7864,
    0.7675, 0.9398, 0.9143, 0.8027, 0.8119, 0.7690, 0.7657, 0.7412, 0.7762,
    0.7036, 1.0839, 0.7384, 0.8037, 0.7584, 0.7105, 0.7081, 0.7401, 0.7523,
    0.7324, 0.9081, 0.8292, 0.7850, 0.7604, 0.7366, 0.7444, 0.6820, 0.7481,
    0.9251, 1.3846, 0.9633, 0.9883, 1.0441, 0.9626, 0.9690, 0.9027, 0.9692,
    1.0290, 0.8795, 0.8795, 0.9544, 0.8723, 0.8092, 0.8017, 0.8077, 0.7916,
    1.1648, 0.9926, 1.1819, 1.0435, 1.0595, 0.9841, 0.9865, 0.9574, 0.9949,
    0.9948, 0.9310, 1.0751, 0.9465, 0.9913, 0.8504, 0.8539, 0.8482, 0.8740,
    1.0672, 0.7972, 1.0034, 0.9025, 0.8712, 0.9299, 0.9304, 0.9033, 0.9842,
    0.9102, 0.7796, 1.0354, 0.8004, 0.9222, 0.9436, 0.9329, 0.8641, 0.9491,
    0.9254, 0.8628, 0.8732, 0.8247, 0.8396, 0.8087, 0.8056, 0.7499, 0.9135
  ),
  RMSE = c(
    1.1749, 1.0201, 1.1203, 1.0217, 1.0247, 1.0258, 1.0456, 1.0221, 0.9458,
    1.0524, 1.0672, 0.9087, 0.7955, 0.9634, 1.1349, 1.1566, 1.0957, 1.0566,
    0.8927, 0.7836, 1.0107, 0.8931, 0.8347, 0.8721, 0.8675, 0.8363, 0.8163,
    1.0005, 1.2495, 1.1011, 1.2295, 1.0560, 1.0080, 1.0010, 1.0240, 1.0151,
    1.0906, 0.9525, 1.1011, 0.9345, 0.9435, 0.9160, 0.9183, 0.8933, 1.0316,
    0.9684, 0.8041, 0.8801, 0.8355, 1.08, 1.8547, 1.8542, 2.0369, 1.0165,
    0.9662, 1.1820, 1.1200, 0.9987, 0.9928, 0.9180, 0.9149, 0.9052, 0.9440,
    0.8696, 1.4117, 0.9346, 1.0360, 0.9356, 0.9108, 0.9054, 0.9524, 0.9640,
    0.8822, 1.1042, 1.0348, 0.9886, 0.9349, 0.8874, 0.8945, 0.8441, 0.9048,
    1.1538, 1.8206, 1.1530, 1.1878, 1.2549, 1.1382, 1.1430, 1.0991, 1.1418,
    1.1922, 1.1146, 1.1146, 1.1698, 1.0844, 1.0218, 1.0142, 1.0153, 1.0016,
    1.4132, 1.2913, 1.4575, 1.3605, 1.2953, 1.2073, 1.2136, 1.1797, 1.2255,
    1.2155, 1.1429, 1.3151, 1.2140, 1.2291, 1.0716, 1.0767, 1.0782, 1.1069,
    1.2802, 0.9947, 1.2714, 1.1492, 1.0795, 1.1870, 1.1872, 1.1593, 1.2405,
    1.0543, 0.9559, 1.2643, 1.0040, 1.1237, 1.1514, 1.1379, 1.0603, 1.1609,
    1.1305, 1.0794, 1.0835, 1.1614, 1.0545, 1.0066, 1.0011, 0.9494, 1.1162
  )
)

# Calculate average MAE and RMSE per station
avg_metrics <- metrics_data %>%
  group_by(Station) %>%
  summarize(
    Avg_MAE = mean(MAE),
    Avg_RMSE = mean(RMSE)
  ) %>%
  pivot_longer(cols = c(Avg_MAE, Avg_RMSE), names_to = "Metric", values_to = "Value")

# Create horizontal bar plot
ggplot(avg_metrics, aes(x = Value, y = Station, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values = c("Avg_MAE" = "#1f77b4", "Avg_RMSE" = "#ff7f0e"),
                    labels = c("MAE", "RMSE")) +
  labs(
    title = "Average Model Performance Comparison",
    x = "Error Value",
    y = "Station",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 10),
    axis.text.x = element_text(size = 8),
    legend.position = "top",
    panel.grid.major.y = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
  geom_text(aes(label = round(Value, 2)), 
            position = position_dodge(width = 0.7),
            hjust = -0.2, size = 3)







