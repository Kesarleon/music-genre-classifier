import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def create_dataset(n_samples=5000):
    """
    Generates a synthetic and imbalanced dataset for music genre classification.
    """
    np.random.seed(42)
    generos = ["Pop", "Rock", "Reggaeton", "Jazz", "Clasica"]
    probs = [0.4, 0.3, 0.2, 0.07, 0.03]  # imbalanced distribution

    duracion = np.random.normal(200, 50, n_samples)
    bpm = np.random.normal(120, 30, n_samples)
    energia = np.random.uniform(0, 1, n_samples)
    acustica = np.random.uniform(0, 1, n_samples)
    popularidad = np.random.randint(0, 101, n_samples)
    instrumentalidad = np.random.beta(2, 5, n_samples)

    genero = np.random.choice(generos, size=n_samples, p=probs)

    df = pd.DataFrame({
        "duracion": duracion,
        "bpm": bpm,
        "energia": energia,
        "acustica": acustica,
        "popularidad": popularidad,
        "instrumentalidad": instrumentalidad,
        "genero_musical": genero
    })
    return df

def build_pipeline():
    """
    Builds the preprocessing and modeling pipeline using StandardScaler, SMOTE, and XGBClassifier.
    """
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ])
    return pipeline

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn and saves it to a file.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix - Tuned XGBoost")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix.png")
    print("\nConfusion matrix saved to confusion_matrix.png")

if __name__ == '__main__':
    # --- Data Preparation ---
    df = create_dataset()
    print("Synthetic dataset created.")
    print(df["genero_musical"].value_counts(normalize=True))
    print("-" * 30)

    X = df.drop(columns=["genero_musical"])
    y = df["genero_musical"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print("Data split into training and testing sets.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("-" * 30)

    # --- Pipeline and Hyperparameter Tuning ---
    pipeline = build_pipeline()
    print("Pipeline created:")
    print(pipeline)
    print("-" * 30)

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.1, 0.01],
        'classifier__subsample': [0.7, 1.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1
    )

    print("Starting hyperparameter tuning with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print("-" * 30)

    # --- Final Evaluation on the Test Set ---
    print("Evaluating the best model on the test set...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names=le.classes_)
