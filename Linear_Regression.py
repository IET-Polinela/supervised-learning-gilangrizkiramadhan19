import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Load Dataset --------------------
# Dataset dengan outlier
df_outlier = pd.read_csv("train_with_outliers.csv")

# Dataset tanpa outlier - StandardScaler
df_std_scaled = pd.read_csv("train_no_outliers_standard_scaled.csv")

# Dataset tanpa outlier - MinMaxScaler
df_minmax_scaled = pd.read_csv("train_no_outliers_minmax_scaled.csv")

# -------------------- Fitur dan Target --------------------
features = ["GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "OverallQual", "YearBuilt"]
target = "SalePrice"

# -------------------- Pisahkan Fitur dan Target --------------------
X_outlier = df_outlier[features]
y_outlier = df_outlier[target]

X_std = df_std_scaled[features]
y_std = df_std_scaled[target]

X_minmax = df_minmax_scaled[features]
y_minmax = df_minmax_scaled[target]

# -------------------- Split Data --------------------
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_outlier, y_outlier, test_size=0.2, random_state=42)
X_train_std, X_test_std, y_train_std, y_test_std = train_test_split(X_std, y_std, test_size=0.2, random_state=42)
X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = train_test_split(X_minmax, y_minmax, test_size=0.2, random_state=42)

# -------------------- Inisialisasi dan Training Model --------------------
model_outlier = LinearRegression()
model_std = LinearRegression()
model_minmax = LinearRegression()

model_outlier.fit(X_train_out, y_train_out)
model_std.fit(X_train_std, y_train_std)
model_minmax.fit(X_train_minmax, y_train_minmax)

# -------------------- Prediksi --------------------
y_pred_out = model_outlier.predict(X_test_out)
y_pred_std = model_std.predict(X_test_std)
y_pred_minmax = model_minmax.predict(X_test_minmax)

# -------------------- Evaluasi --------------------
mse_out = mean_squared_error(y_test_out, y_pred_out)
r2_out = r2_score(y_test_out, y_pred_out)

mse_std = mean_squared_error(y_test_std, y_pred_std)
r2_std = r2_score(y_test_std, y_pred_std)

mse_minmax = mean_squared_error(y_test_minmax, y_pred_minmax)
r2_minmax = r2_score(y_test_minmax, y_pred_minmax)

print("Hasil Evaluasi Model Linear Regression:")
print(f"Model dengan Outlier                       - MSE: {mse_out:.2f}, R2 Score: {r2_out:.4f}")
print(f"Model tanpa Outlier (Standard Scaled)      - MSE: {mse_std:.2f}, R2 Score: {r2_std:.4f}")
print(f"Model tanpa Outlier (MinMax Scaled)        - MSE: {mse_minmax:.2f}, R2 Score: {r2_minmax:.4f}")

# -------------------- Direktori Output Visualisasi --------------------
output_dir = "hasil_visualisasi_regresi"
os.makedirs(output_dir, exist_ok=True)

# -------------------- Fungsi Visualisasi Diagnostik --------------------
def plot_model_diagnostics(y_test, y_pred, model_name, filename):
    residuals = y_test - y_pred
    plt.figure(figsize=(15, 5))

    # Scatter Plot
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Harga Aktual")
    plt.ylabel("Harga Prediksi")
    plt.title(f"{model_name} - Scatter Plot")

    # Residual Plot
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.5, color='orange')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel("Harga Prediksi")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} - Residual Plot")

    # Distribusi Residual
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, bins=30, color="navy")
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title(f"{model_name} - Distribusi Residual")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.show()

# -------------------- Visualisasi Semua Model --------------------
plot_model_diagnostics(y_test_out, y_pred_out, "Regresi (Dengan Outlier)", "regresi_dengan_outlier.png")
plot_model_diagnostics(y_test_std, y_pred_std, "Regresi (Tanpa Outlier - Standard Scaled)", "regresi_tanpa_outlier_std.png")
plot_model_diagnostics(y_test_minmax, y_pred_minmax, "Regresi (Tanpa Outlier - MinMax Scaled)", "regresi_tanpa_outlier_minmax.png")
