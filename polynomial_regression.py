import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dataset dengan outlier
df_outlier = pd.read_csv("train_with_outliers.csv")

# Dataset tanpa outlier - StandardScaler
df_std_scaled = pd.read_csv("train_no_outliers_standard_scaled.csv")

# Dataset tanpa outlier - MinMaxScaler
df_minmax_scaled = pd.read_csv("train_no_outliers_minmax_scaled.csv")

# Gabungkan semua dataset dalam dict
datasets = {
    "Dengan Outlier": df_outlier,
    "StandardScaler": df_std_scaled,
    "MinMaxScaler": df_minmax_scaled
}

# Fitur yang digunakan
features = ["GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "OverallQual", "YearBuilt"]
target = 'SalePrice'

# Buat folder output
output_dir = "visualisasi_polynomial"
os.makedirs(output_dir, exist_ok=True)

# Fungsi visualisasi
def visualisasi_perbandingan(y_test, y_preds, label_models, dataset_name):
    plt.figure(figsize=(10, 6))
    for y_pred, label in zip(y_preds, label_models):
        plt.scatter(y_test, y_pred, alpha=0.5, label=label)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Prediksi vs Aktual - {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"prediksi_vs_aktual_{dataset_name}.png"))
    plt.close()

    # Plot residuals
    plt.figure(figsize=(15, 4))
    for i, (y_pred, label) in enumerate(zip(y_preds, label_models)):
        plt.subplot(1, 3, i + 1)
        sns.histplot(y_test - y_pred, kde=True, bins=30)
        plt.title(f"Residual {label}")
        plt.xlabel("Residuals")
    plt.suptitle(f"Distribusi Residual - {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"residuals_{dataset_name}.png"))
    plt.close()

# Loop untuk setiap dataset
for nama_dataset, df in datasets.items():
    print(f"\n=== {nama_dataset} ===")
    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    model_lin = LinearRegression()
    model_lin.fit(X_train, y_train)
    y_pred_lin = model_lin.predict(X_test)
    mse_lin = mean_squared_error(y_test, y_pred_lin)
    r2_lin = r2_score(y_test, y_pred_lin)

    # Polynomial Degree 2
    poly2 = PolynomialFeatures(degree=2)
    X_train_poly2 = poly2.fit_transform(X_train)
    X_test_poly2 = poly2.transform(X_test)
    model_poly2 = LinearRegression()
    model_poly2.fit(X_train_poly2, y_train)
    y_pred_poly2 = model_poly2.predict(X_test_poly2)
    mse_poly2 = mean_squared_error(y_test, y_pred_poly2)
    r2_poly2 = r2_score(y_test, y_pred_poly2)

    # Polynomial Degree 3
    poly3 = PolynomialFeatures(degree=3)
    X_train_poly3 = poly3.fit_transform(X_train)
    X_test_poly3 = poly3.transform(X_test)
    model_poly3 = LinearRegression()
    model_poly3.fit(X_train_poly3, y_train)
    y_pred_poly3 = model_poly3.predict(X_test_poly3)
    mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
    r2_poly3 = r2_score(y_test, y_pred_poly3)

    # Print hasil evaluasi
    print(f"Linear Regression     - MSE: {mse_lin:.2f}, R2: {r2_lin:.4f}")
    print(f"Polynomial (Deg=2)    - MSE: {mse_poly2:.2f}, R2: {r2_poly2:.4f}")
    print(f"Polynomial (Deg=3)    - MSE: {mse_poly3:.2f}, R2: {r2_poly3:.4f}")

    # Visualisasi hasil
    visualisasi_perbandingan(
        y_test,
        [y_pred_lin, y_pred_poly2, y_pred_poly3],
        ['Linear', 'Polynomial D=2', 'Polynomial D=3'],
        nama_dataset
    )
