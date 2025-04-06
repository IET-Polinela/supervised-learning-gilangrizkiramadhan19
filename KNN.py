import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Buat folder visualisasi jika belum ada
output_dir = "visualisasi_knn"
os.makedirs(output_dir, exist_ok=True)

# Dataset yang digunakan
datasets = {
    "with_outliers": {
        "data": pd.read_csv("train_with_outliers.csv"),
        "scaler": StandardScaler(),
        "name": "Dengan Outlier"
    },
    "no_outliers_std": {
        "data": pd.read_csv("train_no_outliers_standard_scaled.csv"),
        "scaler": None,
        "name": "StandardScaler"
    },
    "no_outliers_minmax": {
        "data": pd.read_csv("train_no_outliers_minmax_scaled.csv"),
        "scaler": None,
        "name": "MinMaxScaler"
    }
}

# Nilai K untuk KNN
k_values = [3, 5, 7]
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

# Loop setiap dataset
for key, dataset in datasets.items():
    print(f"\n=== {dataset['name']} ===")
    df = dataset["data"]
    scaler = dataset["scaler"]

    X = df[["GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "OverallQual", "YearBuilt"]]
    y = df['SalePrice']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scaling jika perlu
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values

    results = []

    # Visualisasi Prediksi vs Aktual untuk KNN
    plt.figure(figsize=(18, 5))
    for i, k in enumerate(k_values):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append((f'KNN (K={k})', mse, r2, y_pred))

        plt.subplot(1, 3, i+1)
        sns.scatterplot(x=y_test, y=y_pred, color=colors[i], alpha=0.6, edgecolor='black')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel("Harga Aktual")
        plt.ylabel("Harga Prediksi")
        plt.title(f"K = {k}\nMSE: {mse:.2f} | R2: {r2:.2f}")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediksi_vs_aktual_{key}.png")
    plt.show()

    # Barplot R2 Score (fix warning)
    plt.figure(figsize=(8, 6))
    r2_scores = [r2 for _, _, r2, _ in results]
    labels = [label for label, _, _, _ in results]
    sns.barplot(x=labels, y=r2_scores, hue=labels, palette=colors, legend=False)
    plt.title(f'R2 Score untuk {dataset["name"]}')
    plt.ylabel("R2 Score")
    plt.ylim(0, 1)
    plt.savefig(f"{output_dir}/r2_barplot_{key}.png")
    plt.show()

    # Residual Boxplot
    plt.figure(figsize=(18, 5))
    for i, (_, _, _, y_pred) in enumerate(results):
        residuals = y_test.values - y_pred
        plt.subplot(1, 3, i+1)
        sns.boxplot(y=residuals, color=colors[i])
        plt.title(f'Boxplot Residual ({results[i][0]})')
        plt.ylabel("Residual")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/residual_boxplot_{key}.png")
    plt.show()

    # Evaluasi Model Tambahan
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    mse_lin = mean_squared_error(y_test, y_pred_lin)
    r2_lin = r2_score(y_test, y_pred_lin)

    # Polynomial Regression Degree 2
    poly2 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly2.fit(X_train, y_train)
    y_pred_poly2 = poly2.predict(X_test)
    mse_poly2 = mean_squared_error(y_test, y_pred_poly2)
    r2_poly2 = r2_score(y_test, y_pred_poly2)

    # Polynomial Regression Degree 3
    poly3 = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly3.fit(X_train, y_train)
    y_pred_poly3 = poly3.predict(X_test)
    mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
    r2_poly3 = r2_score(y_test, y_pred_poly3)

    # Tampilkan hasil evaluasi
    for label, mse, r2, _ in results:
        print(f"{label:<20} - MSE: {mse:.2f}, R2: {r2:.4f}")
    print(f"Linear Regression     - MSE: {mse_lin:.2f}, R2: {r2_lin:.4f}")
    print(f"Polynomial (Deg=2)    - MSE: {mse_poly2:.2f}, R2: {r2_poly2:.4f}")
    print(f"Polynomial (Deg=3)    - MSE: {mse_poly3:.2f}, R2: {r2_poly3:.4f}")
