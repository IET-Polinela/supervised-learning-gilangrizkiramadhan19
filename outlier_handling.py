import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Menentukan path file dataset
file_path = "train.csv"
df = pd.read_csv(file_path)

# Membuat folder untuk menyimpan hasil visualisasi
output_dir = "visualisasi_outlier"
os.makedirs(output_dir, exist_ok=True)

# 1. Menampilkan jumlah data awal
print(f"Jumlah data sebelum menghapus outlier: {df.shape[0]}")

# 2. Menampilkan statistik deskriptif sebelum menghapus outlier
print("\nStatistik Sebelum Handling Outlier:")
print(df.describe())

# 3. Identifikasi Outlier menggunakan Metode IQR (Interquartile Range)
def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Mendeteksi outlier untuk setiap fitur numerik
outliers_mask = df.select_dtypes(include=[np.number]).apply(detect_outliers_iqr)

# 4. Visualisasi boxplot sebelum menangani outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title("Boxplot Sebelum Handling Outlier")
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, "boxplot_sebelum_outlier.png"))  # Simpan gambar
plt.close()

# 5. Menghapus Outlier berdasarkan Metode IQR
df_no_outliers = df[~outliers_mask.any(axis=1)]

# 6. Menampilkan jumlah data setelah penghapusan outlier
print(f"\nJumlah data setelah menghapus outlier: {df_no_outliers.shape[0]}")

# 7. Menampilkan statistik deskriptif setelah menghapus outlier
print("\nStatistik Setelah Handling Outlier:")
print(df_no_outliers.describe())

# 8. Visualisasi boxplot setelah menangani outlier
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_no_outliers)
plt.title("Boxplot Setelah Handling Outlier")
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, "boxplot_setelah_outlier.png"))  # Simpan gambar
plt.close()

# 9. Fokus pada fitur SalePrice sebelum dan sesudah outlier removal
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['SalePrice'])
plt.title("Boxplot SalePrice Sebelum Handling Outlier")
plt.savefig(os.path.join(output_dir, "boxplot_SalePrice_sebelum.png"))  # Simpan gambar
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(y=df_no_outliers['SalePrice'])
plt.title("Boxplot SalePrice Setelah Handling Outlier")
plt.savefig(os.path.join(output_dir, "boxplot_SalePrice_setelah.png"))  # Simpan gambar
plt.close()

# 10. Scatter Plot sebelum menangani outlier
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'], color='red', edgecolor='white')
plt.title("Scatter Plot Sebelum Handling Outlier")
plt.xlabel("GrLivArea (Luas Bangunan)")
plt.ylabel("SalePrice (Harga Rumah)")
plt.savefig(os.path.join(output_dir, "scatter_sebelum_outlier.png"))  # Simpan gambar
plt.close()

# 11. Scatter Plot setelah menangani outlier
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_no_outliers['GrLivArea'], y=df_no_outliers['SalePrice'], color='blue', edgecolor='white')
plt.title("Scatter Plot Setelah Handling Outlier")
plt.xlabel("GrLivArea (Luas Bangunan)")
plt.ylabel("SalePrice (Harga Rumah)")
plt.savefig(os.path.join(output_dir, "scatter_setelah_outlier.png"))  # Simpan gambar
plt.close()

# 12. Menyimpan dataset dengan dan tanpa outlier
df.to_csv("train_with_outliers.csv", index=False)
df_no_outliers.to_csv("train_no_outliers.csv", index=False)

# 13. Analisis bagaimana outlier mempengaruhi hasil model regresi
print("\nBagaimana Outlier Mempengaruhi Hasil Model Regresi:")
print("- Distorsi koefisien regresi, sehingga prediksi menjadi tidak akurat.")
print("- Model lebih cenderung overfitting terhadap nilai ekstrim.")
print("- Meningkatkan varians dari model yang menyebabkan performa buruk pada data baru.")

# 14. Metode terbaik untuk menangani outlier
print("\nMetode Terbaik dalam Menangani Outlier:")
print("- Metode IQR lebih robust terhadap distribusi data yang tidak normal.")
print("- IQR efektif mempertahankan data utama tanpa membuang terlalu banyak informasi.")
print("- Metode ini digunakan secara luas untuk membersihkan data sebelum analisis lebih lanjut.")

print(f"\nSemua hasil visualisasi telah disimpan dalam folder: {output_dir}")
