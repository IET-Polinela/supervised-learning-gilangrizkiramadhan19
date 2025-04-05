import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


file_path = "train.csv"  
df = pd.read_csv(file_path)


print("📌 Informasi Dataset:")
print(df.info())
print("\n📌 Tampilan Awal Data:")
print(df.head())



df.drop_duplicates(inplace=True)


for col in df.select_dtypes(include=['number']).columns:
    df[col].fillna(df[col].median(), inplace=True)  

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)  


print("\n Nilai yang Hilang setelah Imputasi:")
print(df.isnull().sum())


cat_cols = df.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, drop="first")  
encoded_cat = encoder.fit_transform(df[cat_cols])


encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(cat_cols))


df_encoded = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
print("\n📌 Dataset setelah OneHot Encoding:")
print(df_encoded.head())


scaler = StandardScaler()
num_cols = df_encoded.select_dtypes(include=['number']).columns
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])


X = df_encoded.drop(columns=["SalePrice"])  
y = df_encoded["SalePrice"]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


final_df = pd.concat([X_train, X_test, y_train, y_test], axis=1)
final_df.to_csv("train_processed_final.csv", index=False)


print("\n📌 Dataset setelah encoding, normalisasi, dan pemisahan data training & testing disimpan sebagai 'train_processed_final.csv'.")
print("\n📌 Shape Data Training:", X_train.shape, "Shape Data Testing:", X_test.shape)
print("\n📌 Sample Data Training:")
print(X_train.head())
