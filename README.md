# 🎶 Music Genre Classifier MLOps Pipeline

This project implements an end-to-end MLOps pipeline for training a music genre classification model. It uses a synthetic, imbalanced dataset and includes steps for data generation, preprocessing, training (with and without SMOTE), and evaluation.

---

## 🚀 Project Structure

The project is organized into a modular structure to promote reproducibility and scalability:

-   `src/`: Contains the Python scripts for each step of the pipeline.
-   `data/`: Stores the raw and processed datasets.
-   `models/`: Contains the serialized (pickled) trained models and the label encoder.
-   `metrics/`: Stores the evaluation results, such as classification reports and confusion matrices.
-   `notebooks/`: Contains the original exploratory notebook.
-   `tests/`: Contains unit tests for the pipeline components.
-   `requirements.txt`: Lists the required Python dependencies.
-   `run_pipeline.sh`: An orchestration script to run the entire pipeline.

---

## 📦 Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/<your_username>/music-genre-classifier.git
    cd music-genre-classifier
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Run the pipeline:**
    Execute the orchestration script to run all the steps, from data generation to model evaluation. This will install dependencies, generate data, preprocess it, and train/evaluate two models (one with SMOTE and one without).
    ```bash
    bash run_pipeline.sh
    ```

4.  **Check the results:**
    -   The trained models will be in the `models/` directory (`model.pkl` and `model_smote.pkl`).
    -   The evaluation metrics will be in the `metrics/` directory.
    -   A performance dashboard is generated at `dashboard.html`. Open this file in your web browser to see a comparison of the models.

---

## 🧪 Running Tests

To run the unit tests, execute the following command:
```bash
python -m unittest discover tests/
```
