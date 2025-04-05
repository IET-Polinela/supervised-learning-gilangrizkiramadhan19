import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset tanpa outlier
file_path = "train_no_outliers.csv"
df = pd.read_csv(file_path)

# Pilih fitur yang akan divisualisasikan
feature = "SalePrice"

# Scaling
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

df_standard_scaled = df.copy()
df_minmax_scaled = df.copy()

# Lakukan scaling pada fitur 'SalePrice'
df_standard_scaled[feature] = scaler_standard.fit_transform(df[[feature]])
df_minmax_scaled[feature] = scaler_minmax.fit_transform(df[[feature]])

# Buat folder untuk menyimpan gambar jika belum ada
output_dir = "visualisasi_scaling"
os.makedirs(output_dir, exist_ok=True)

# Plot 3 visualisasi dalam satu figure
plt.figure(figsize=(15, 5))

# 1. Histogram sebelum scaling
plt.subplot(1, 3, 1)
sns.histplot(df[feature], kde=True, color="blue", bins=30)
plt.title(f"{feature} Sebelum Scaling")
plt.xlabel("SalePrice")
plt.ylabel("Frekuensi")
plt.grid(True)

# 2. Histogram setelah StandardScaler
plt.subplot(1, 3, 2)
sns.histplot(df_standard_scaled[feature], kde=True, color="red", bins=30)
plt.title(f"{feature} Setelah StandardScaler")
plt.xlabel("SalePrice")
plt.ylabel("Frekuensi")
plt.grid(True)

# 3. Histogram setelah MinMaxScaler
plt.subplot(1, 3, 3)
sns.histplot(df_minmax_scaled[feature], kde=True, color="green", bins=30)
plt.title(f"{feature} Setelah MinMaxScaler")
plt.xlabel("SalePrice")
plt.ylabel("Frekuensi")
plt.grid(True)

plt.tight_layout()

# Simpan gambar ke dalam folder
plot_path = os.path.join(output_dir, "scaling_comparison.png")
plt.savefig(plot_path, dpi=300)

plt.show()

# Simpan dataset hasil scaling ke file CSV
scaled_file_path_standard = "train_no_outliers_standard_scaled.csv"
scaled_file_path_minmax = "train_no_outliers_minmax_scaled.csv"

df_standard_scaled.to_csv(scaled_file_path_standard, index=False)
df_minmax_scaled.to_csv(scaled_file_path_minmax, index=False)

print(f"Dataset hasil StandardScaler disimpan di: {scaled_file_path_standard}")
print(f"Dataset hasil MinMaxScaler disimpan di: {scaled_file_path_minmax}")
