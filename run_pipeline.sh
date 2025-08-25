#!/bin/bash

echo "Starting the MLOps pipeline..."

# Step 1: Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 2: Generate data
echo "Generating synthetic data..."
python src/generate_data.py

# Step 3: Preprocess data
echo "Preprocessing data..."
python src/preprocess.py

# Step 4: Train model without SMOTE
echo "Training model without SMOTE..."
python src/train.py

# Step 5: Evaluate model without SMOTE
echo "Evaluating model without SMOTE..."
python src/evaluate.py --model-name "model.pkl"

# Step 6: Train model with SMOTE
echo "Training model with SMOTE..."
python src/train.py --use-smote

# Step 7: Evaluate model with SMOTE
echo "Evaluating model with SMOTE..."
python src/evaluate.py --model-name "model_smote.pkl"

# Step 8: Generate dashboard
echo "Generating HTML dashboard..."
python src/generate_dashboard.py

echo "Pipeline finished successfully. The dashboard is available at dashboard.html"
