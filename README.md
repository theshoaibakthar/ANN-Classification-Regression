# ANN - Classification (Churn) and Regression (Salary)

Project: A pair of small but complete Artificial Neural Network (ANN) pipelines focused on:

- Churn classification: a binary-class ANN to predict customer churn.
- Salary regression: an ANN to predict salary (regression).

## Contents

- `1_experiments.ipynb` — Primary experimentation notebook for classification model development.
- `2_prediction.ipynb` — Prediction / inference examples for classification.
- `3_salary_regression.ipynb` — Regression notebook for salary prediction.
- `app.py` — Script demonstrating classification inference.
- `app_regression.py` — Script demonstrating regression inference.
- `Churn_Modelling.csv` — The dataset used for classification experiments.
- `requirements.txt` — Python dependencies used in the project.
- `artifacts/` — Saved model artifacts and outputs.
	- `artifacts/classification/classification_model.keras` — trained classification model
	- `artifacts/regression/regression_model.keras` — trained regression model

## Goals and scope

This repo is intended to be a compact, reproducible example of building, training, and serving small ANNs for both classification (churn) and regression (salary).

Key design principles:
- Clear, minimal code and notebooks for reproducibility.
- Saved model artifacts that can be loaded for inference.
- Training logs for quick inspection with TensorBoard.

## Quick start

1. Create and activate a virtual environment (recommended):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```cmd
pip install -r requirements.txt
```

3. Run the classification example app (inference):

```cmd
streamlit run app.py
```

4. Run the regression example app:

```cmd
streamlit run app_regression.py
```

Notes: `app.py` and `app_regression.py` are small runner scripts that demonstrate loading the saved model from `artifacts/` and running inference on sample inputs.

## Loading the saved Keras models (example)

In Python you can load a saved model with TensorFlow / Keras:

```py
from tensorflow.keras.models import load_model

# Classification
clf = load_model(r"artifacts/classification/classification_model.keras")

# Regression
reg = load_model(r"artifacts/regression/regression_model.keras")
```

## Training & experiments

- Notebooks contain the data preparation, model definition, training and evaluation steps.
- TensorBoard logs are in `classification_logs/` and `regression_logs/`; to inspect them:

```cmd
# from repository root
tensorboard --logdir classification_logs\fit --port 6006
```

Adjust `--logdir` to point to the specific training run folder you want to inspect (e.g. the timestamped run). Open the printed URL in a browser.

## Contact

For questions, issues, or contributions, open an issue or pull request in the repository.

