import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Load Dataset --------------------
df_outlier = pd.read_csv("train_with_outliers.csv")
df_clean = pd.read_csv("train_no_outliers_standard_scaled.csv")

# -------------------- Fitur dan Target --------------------
features = ["GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF"]
target = "SalePrice"

X_outlier = df_outlier[features]
y_outlier = df_outlier[target]

X_clean = df_clean[features]
y_clean = df_clean[target]

# -------------------- Scaling untuk data bersih --------------------
scaler = StandardScaler()
X_clean_scaled = scaler.fit_transform(X_clean)

# -------------------- Split Data --------------------
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_outlier, y_outlier, test_size=0.2, random_state=42)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean_scaled, y_clean, test_size=0.2, random_state=42)

# -------------------- Inisialisasi dan Training Model --------------------
model_outlier = LinearRegression()
model_clean = LinearRegression()

model_outlier.fit(X_train_out, y_train_out)
model_clean.fit(X_train_clean, y_train_clean)

# -------------------- Prediksi --------------------
y_pred_out = model_outlier.predict(X_test_out)
y_pred_clean = model_clean.predict(X_test_clean)

# -------------------- Evaluasi --------------------
mse_out = mean_squared_error(y_test_out, y_pred_out)
r2_out = r2_score(y_test_out, y_pred_out)

mse_clean = mean_squared_error(y_test_clean, y_pred_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

print("Hasil Evaluasi Model:")
print(f"Model dengan Outlier       - MSE: {mse_out:.2f}, R2 Score: {r2_out:.4f}")
print(f"Model tanpa Outlier (Standard Scaled) - MSE: {mse_clean:.2f}, R2 Score: {r2_clean:.4f}")

# -------------------- Direktori Output Visualisasi --------------------
output_dir = "hasil_visualisasi_regresi"
os.makedirs(output_dir, exist_ok=True)

# -------------------- Fungsi Visualisasi --------------------
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

# -------------------- Visualisasi --------------------
plot_model_diagnostics(y_test_out, y_pred_out, "Regresi (Dengan Outlier)", "regresi_dengan_outlier.png")
plot_model_diagnostics(y_test_clean, y_pred_clean, "Regresi (Tanpa Outlier - Scaled)", "regresi_tanpa_outlier_scaled.png")
