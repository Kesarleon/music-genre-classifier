import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def preprocess_data(input_path="data/raw_music_data.csv", output_dir="data", models_dir="models"):
    """
    Loads raw data, preprocesses it, and splits it into training and test sets.
    """
    df = pd.read_csv(input_path)

    X = df.drop(columns=["genero_musical"])
    y = df["genero_musical"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save the processed data
    # We need to reset index to concatenate correctly
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    train_df = pd.concat([X_train, pd.DataFrame(y_train, columns=['genero_musical'])], axis=1)
    test_df = pd.concat([X_test, pd.DataFrame(y_test, columns=['genero_musical'])], axis=1)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    # Save the label encoder
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("Data preprocessed and saved to data/train.csv and data/test.csv")
    print("Label encoder saved to models/label_encoder.pkl")

if __name__ == "__main__":
    if not os.path.exists("data/raw_music_data.csv"):
        print("Raw data not found. Please run generate_data.py first.")
    else:
        preprocess_data()
