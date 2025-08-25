import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os
import pickle
import argparse

def train_model(data_path="data/train.csv", models_dir="models", use_smote=False):
    """
    Loads preprocessed data, trains a model, and saves it.
    """
    df_train = pd.read_csv(data_path)

    X_train = df_train.drop(columns=["genero_musical"])
    y_train = df_train["genero_musical"]

    if use_smote:
        print("Using SMOTE for balancing the training data.")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        model_filename = "model_smote.pkl"
    else:
        print("Training without SMOTE.")
        model_filename = "model.pkl"

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a music genre classifier.")
    parser.add_argument("--use-smote", action="store_true", help="Use SMOTE to balance training data.")
    args = parser.parse_args()

    if not os.path.exists("data/train.csv"):
        print("Training data not found. Please run preprocess.py first.")
    else:
        train_model(use_smote=args.use_smote)
