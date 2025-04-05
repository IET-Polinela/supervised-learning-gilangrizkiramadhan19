import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Pastikan folder visualisasi tersedia
output_dir = "visualisasi_knn"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = pd.read_csv('train_no_outliers_standard_scaled.csv')

# Pilih fitur dan target (TAMBAH FITUR agar R2 berubah)
X = data[['GrLivArea', 'OverallQual', 'GarageArea', 'YearBuilt']]
y = data['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Regression untuk beberapa nilai K
k_values = [3, 5, 7]
results = []
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

# Visualisasi Prediksi vs Aktual
plt.figure(figsize=(18, 5))
for i, k in enumerate(k_values):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((k, mse, r2, y_pred))

    plt.subplot(1, 3, i+1)
    sns.scatterplot(x=y_test, y=y_pred, color=colors[i], alpha=0.6, edgecolor='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('Harga Aktual')
    plt.ylabel('Harga Prediksi')
    plt.title(f'K = {k}\nMSE: {mse:.2f} | R2: {r2:.2f}')

plt.tight_layout()
plt.savefig(f'{output_dir}/prediksi_vs_aktual_modif.png')
plt.show()

# Cetak hasil evaluasi
print("\nHasil Evaluasi KNN Regression:")
for k, mse, r2, _ in results:
    print(f"K = {k} | MSE: {mse:.2f} | R2 Score: {r2:.2f}")

# Bar Plot R2 Score
plt.figure(figsize=(8, 6))
r2_scores = [r2 for _, _, r2, _ in results]
sns.barplot(x=[f'K={k}' for k, _, _, _ in results], y=r2_scores, palette=colors)
plt.title('Perbandingan R2 Score untuk Tiap Nilai K')
plt.ylabel('R2 Score')
plt.ylim(0, 1)
plt.savefig(f'{output_dir}/r2_comparison_barplot.png')
plt.show()

# Residual Plot dengan Boxplot
plt.figure(figsize=(18, 5))
for i, (k, _, _, y_pred) in enumerate(results):
    residuals = y_test.values - y_pred
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=residuals, color=colors[i])
    plt.title(f'Boxplot Residual (K = {k})')
    plt.ylabel('Residuals')

plt.tight_layout()
plt.savefig(f'{output_dir}/residual_boxplot_modif.png')
plt.show()
