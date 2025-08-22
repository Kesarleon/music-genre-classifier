import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def create_dataset(n_samples=5000):
    """
    Generates a more complex synthetic and imbalanced dataset for music genre classification.
    """
    np.random.seed(42)
    generos = ["Pop", "Rock", "Reggaeton", "Jazz", "Clasica"]
    probs = [0.4, 0.3, 0.2, 0.07, 0.03]

    countries = ["USA", "UK", "Spain", "Colombia", "Germany"]
    country_probs = [0.4, 0.3, 0.15, 0.1, 0.05]

    duracion = np.random.normal(200, 50, n_samples)
    bpm = np.random.normal(120, 30, n_samples)
    energia = np.random.uniform(0, 1, n_samples)
    acustica = np.random.uniform(0, 1, n_samples)
    popularidad = np.random.randint(0, 101, n_samples)
    instrumentalidad = np.random.beta(2, 5, n_samples)

    start_date = pd.to_datetime("1980-01-01")
    end_date = pd.to_datetime("2023-12-31")
    time_delta = (end_date - start_date).total_seconds()
    random_seconds = np.random.uniform(0, time_delta, n_samples)
    release_date = start_date + pd.to_timedelta(random_seconds, unit='s')

    country_of_origin = np.random.choice(countries, size=n_samples, p=country_probs)
    bpm_energy_interaction = bpm * energia

    genero = np.random.choice(generos, size=n_samples, p=probs)

    df = pd.DataFrame({
        "duracion": duracion,
        "bpm": bpm,
        "energia": energia,
        "acustica": acustica,
        "popularidad": popularidad,
        "instrumentalidad": instrumentalidad,
        "release_date": release_date,
        "country_of_origin": country_of_origin,
        "bpm_energy_interaction": bpm_energy_interaction,
        "genero_musical": genero
    })

    df['release_date'] = df['release_date'].dt.strftime('%Y-%m-%d')

    return df

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        date_col = pd.to_datetime(X.iloc[:, 0])
        return pd.DataFrame({
            'year': date_col.dt.year,
            'month': date_col.dt.month,
            'day_of_week': date_col.dt.dayofweek
        })

def build_feature_engineering_pipeline():
    """
    Builds the feature engineering pipeline using ColumnTransformer.
    """
    numerical_features = [
        'duracion', 'bpm', 'energia', 'acustica',
        'popularidad', 'instrumentalidad', 'bpm_energy_interaction'
    ]
    categorical_features = ['country_of_origin']
    date_features = ['release_date']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('date', DateFeatureExtractor(), date_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

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
    plt.title("Confusion Matrix - Tuned XGBoost with Feature Engineering")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("confusion_matrix_engineered.png")
    print("\nConfusion matrix saved to confusion_matrix_engineered.png")

if __name__ == '__main__':
    # --- Data Preparation ---
    df = create_dataset()
    print("Synthetic dataset created with new features.")
    print(df.head())
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

    # --- Feature Engineering and Modeling Pipeline ---
    feature_engineering_pipeline = build_feature_engineering_pipeline()

    main_pipeline = ImbPipeline([
        ('preprocessor', feature_engineering_pipeline),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))
    ])

    print("Main pipeline with feature engineering created.")
    print("-" * 30)

    # --- Hyperparameter Tuning ---
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.05],
        'preprocessor__num__with_mean': [True, False] # Example of tuning a preprocessor step
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=main_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1
    )

    print("Starting hyperparameter tuning with GridSearchCV on the full pipeline...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print("-" * 30)

    # --- Final Evaluation ---
    print("Evaluating the best model on the test set...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plot_confusion_matrix(y_test, y_pred, class_names=le.classes_)
