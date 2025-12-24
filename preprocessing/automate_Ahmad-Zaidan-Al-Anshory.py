import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump
import os
from preprocess_module import preprocess_data

def preprocess_data(data, target, save_path, header_path, csv_path):
    features = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    features.remove(target)

    # Simpan header
    pd.DataFrame(columns=features).to_csv(header_path, index=False)
    
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

    # Simpan scaler
    dump(scaler, save_path)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load data
    data = pd.read_csv("preprocessing/foodspoiled_raw.csv")

    # Panggil fungsi dan simpan CSV + scaler
    X_train, X_test, y_train, y_test = preprocess_data(
        data,
        target="label",  # ganti sesuai kolom target
        save_path="preprocessing/scaler.joblib",
        header_path="preprocessing/header.csv",
        csv_path="preprocessing/foodspoiled_preprocessing.csv"
    )

    print("Preprocessing selesai ✅")