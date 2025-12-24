import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import os

def preprocess_data(data, target, csv_path):
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    features = [col for col in numeric_cols if col != target]

    if len(features) == 0:
        raise ValueError("Tidak ada fitur numerik selain target")
    
    # Duplikat
    data = data.drop_duplicates()

    # Missing
    data[features] = data[features].fillna(data[features].mean())

    # Outlier
    for col in features:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]

    # Standarisasi
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])

    # Simpan CSV hasil preprocessing
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    data.to_csv(csv_path, index=False)

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("foodspoiled_raw.csv", sep=';')

    # Panggil fungsi dan simpan CSV
    X_train, X_test, y_train, y_test = preprocess_data(
        data,
        target="Status",
        csv_path="preprocessing/foodspoiled_preprocessing.csv"
    )

    print("Preprocessing selesai ✅")