import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Buat folder untuk menyimpan visualisasi
output_dir = "visualisasi_regresi"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = pd.read_csv('train_no_outliers_standard_scaled.csv')

# Pilih fitur dan target
features = ['GrLivArea', 'LotArea', 'TotalBsmtSF', '1stFlrSF']
target = 'SalePrice'
X = data[features]
y = data[target]

# Bagi data menjadi train dan test set dengan random_state yang berbeda
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=24  # diganti agar hasil berbeda
)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Polynomial Regression dengan degree 2
poly2 = PolynomialFeatures(degree=2)
X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

poly_reg2 = LinearRegression()
poly_reg2.fit(X_train_poly2, y_train)
y_pred_poly2 = poly_reg2.predict(X_test_poly2)
mse_poly2 = mean_squared_error(y_test, y_pred_poly2)
r2_poly2 = r2_score(y_test, y_pred_poly2)

# Polynomial Regression dengan degree 3
poly3 = PolynomialFeatures(degree=3)
X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

print("Jumlah fitur hasil transformasi Polynomial Degree 3:", X_train_poly3.shape[1])

poly_reg3 = LinearRegression()
poly_reg3.fit(X_train_poly3, y_train)
y_pred_poly3 = poly_reg3.predict(X_test_poly3)
mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
r2_poly3 = r2_score(y_test, y_pred_poly3)

# Cetak hasil evaluasi
print(f'Linear Regression -> MSE: {mse_lin:.2f}, R2: {r2_lin:.4f}')
print(f'Polynomial Regression (Degree 2) -> MSE: {mse_poly2:.2f}, R2: {r2_poly2:.4f}')
print(f'Polynomial Regression (Degree 3) -> MSE: {mse_poly3:.2f}, R2: {r2_poly3:.4f}')

# Visualisasi: Scatter plot hasil prediksi vs aktual
plt.figure(figsize=(12, 5))
plt.scatter(y_test, y_pred_lin, label='Linear Regression', alpha=0.5, color='blue')
plt.scatter(y_test, y_pred_poly2, label='Polynomial Degree 2', alpha=0.5, color='green')
plt.scatter(y_test, y_pred_poly3, label='Polynomial Degree 3', alpha=0.5, marker='x', color='red')  # beda marker
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.title('Comparison of Regression Models')
plt.savefig(os.path.join(output_dir, 'comparison_regression_models.png'))
plt.close()

# Visualisasi: Distribusi residual untuk Polynomial Degree 2 dan 3
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(y_test - y_pred_poly2, bins=30, kde=True, color='green')
plt.title('Residual Distribution - Polynomial Degree 2')
plt.xlabel('Residuals')

plt.subplot(1, 2, 2)
sns.histplot(y_test - y_pred_poly3, bins=30, kde=True, color='red')
plt.title('Residual Distribution - Polynomial Degree 3')
plt.xlabel('Residuals')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'residual_distribution_polynomial.png'))
plt.close()
