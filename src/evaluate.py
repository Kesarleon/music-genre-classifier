import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(test_data_path="data/test.csv", models_dir="models", metrics_dir="metrics", model_filename="model.pkl"):
    """
    Loads a trained model and test data, evaluates the model, and saves the metrics.
    """
    df_test = pd.read_csv(test_data_path)

    X_test = df_test.drop(columns=["genero_musical"])
    y_test = df_test["genero_musical"]

    # Load the label encoder
    with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    # Load the model
    model_path = os.path.join(models_dir, model_filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_)

    # Create metrics directory if it doesn't exist
    os.makedirs(metrics_dir, exist_ok=True)

    report_suffix = "_smote" if "smote" in model_filename else ""

    report_path = os.path.join(metrics_dir, f"classification_report{report_suffix}.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"Confusion Matrix ({'with SMOTE' if 'smote' in model_filename else 'without SMOTE'})")

    cm_path = os.path.join(metrics_dir, f"confusion_matrix{report_suffix}.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained music genre classifier.")
    parser.add_argument("--model-name", type=str, default="model.pkl", help="Name of the model file to evaluate (e.g., model.pkl or model_smote.pkl).")
    args = parser.parse_args()

    if not os.path.exists("data/test.csv"):
        print("Test data not found. Please run preprocess.py first.")
    elif not os.path.exists(os.path.join("models", args.model_name)):
         print(f"Model {args.model_name} not found. Please run train.py first.")
    else:
        evaluate_model(model_filename=args.model_name)
