import pandas as pd


file_path = "train.csv"  
df = pd.read_csv(file_path)


print("📌 Informasi Dataset:")
print(df.info())
print("\n📌 Tampilan Awal Data:")
print(df.head())


numerical_df = df.select_dtypes(include=['number'])


stats_summary = numerical_df.describe().T
stats_summary["Q1"] = numerical_df.quantile(0.25)
stats_summary["Q2"] = numerical_df.median()
stats_summary["Q3"] = numerical_df.quantile(0.75)


print("\n📌 Statistik Deskriptif:")
print(stats_summary)


missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("\n📌 Nilai yang Hilang:")
print(missing_values)


data_types = df.dtypes
print("\n📌 Tipe Data:")
print(data_types)


summary_data = pd.DataFrame({
    'Informasi Dataset': [str(df.info())],
    'Statistik Deskriptif': [stats_summary.to_string()],
    'Nilai yang Hilang': [missing_values.to_string()],
    'Tipe Data': [data_types.to_string()]
})

summary_data.to_csv("Data_Understanding.csv", index=False)

print("\n Semua informasi telah disimpan dalam file 'Data_Understanding.csv'.")
