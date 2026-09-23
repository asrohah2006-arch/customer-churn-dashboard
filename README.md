# Customer Churn Prediction Using ML and DL

A complete, beginner-friendly portfolio project that predicts telecom customer churn and compares two classical machine-learning models with one deep-learning model.

## Why this project

It connects data science to a clear business question: which customer profiles are associated with churn? The code shows the full workflow rather than only model training: cleaning, EDA, reusable preprocessing, fair evaluation, explainability and a dashboard.

## Models

- Logistic Regression: interpretable baseline
- Random Forest: nonlinear ensemble model
- TensorFlow/Keras neural network: deep-learning comparison

Every model uses the same random seed, stratified 80/20 train-test split and untouched test set. At a 0.50 threshold, the comparison reports accuracy, precision, recall, F1 and ROC-AUC, plus confusion matrices and ROC curves. No result is hard-coded.

## Dataset

This package includes `data/raw/Telco-Customer-Churn.csv`, a public telecom churn sample from IBM's repository:
https://github.com/IBM/telco-customer-churn-on-icp4d/tree/master/data

IBM's related customer-churn code pattern:
https://github.com/IBM/customer-churn-prediction

The file has 7,043 customer rows and 21 columns. `Churn` is the target. Review the source terms before redistributing the data separately.

To retrieve a fresh copy:

```bash
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  -o data/raw/Telco-Customer-Churn.csv
```

## Quick start

**Python 3.11 is required.** TensorFlow, the deep-learning library used here, does not support newer versions such as Python 3.13 or 3.14 yet. You do not need Visual Studio or any C++ compiler: every package installs from a pre-built wheel.

### Windows (easiest)

1. Extract this folder anywhere, for example `Documents\Customer_Churn_ML_DL_Project`.
2. Double-click `run_project.bat`.
3. The script finds Python 3.11 or installs it with the official Python install manager, creates the virtual environment, installs the packages, trains the models and starts the dashboard.
4. If a window already failed with an older Python (for example a Meson "Unknown compiler(s)" error), just run the new `run_project.bat` again: it deletes the broken `.venv` and rebuilds it with Python 3.11.

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### macOS / Linux

```bash
./run_project.sh
```

### Manual setup (any OS)

Make sure `python --version` prints 3.11.x first, then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary :all: -r requirements-train.txt

python src/eda.py
python src/train.py --epochs 80
streamlit run app.py
```

### If something fails

- "Could not find or install Python 3.11": install the Python install manager from https://www.python.org/downloads/ (or the Microsoft Store), then run `run_project.bat` again.
- The window stays open on every failure and prints the step that stopped, so you can read the message before closing it.

## Outputs created after training

- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `models/ann_preprocessor.joblib`
- `models/ann.keras`
- `reports/metrics.csv` and `metrics.json`
- `reports/roc_curves.png`
- `reports/confusion_matrices.png`
- `reports/permutation_importance.csv`
- `reports/shap_importance.csv` when SHAP succeeds

Permutation importance always uses at most 1,000 held-out rows with five repeats, so the fallback remains bounded. SHAP is attempted on up to 250 held-out rows.

## Project structure

```text
customer-churn-ml-dl/
├── app.py
├── README.md
├── RESUME_PROJECT.md
├── requirements.txt        (dashboard only, used by Streamlit Cloud)
├── requirements-train.txt  (full training stack)
├── data/raw/Telco-Customer-Churn.csv
├── models/
├── reports/
├── src/
│   ├── data.py
│   ├── eda.py
│   ├── evaluate.py
│   └── train.py
└── tests/test_data.py
```

## Test

```bash
pytest -q
```

## How to explain it in an interview

1. Churn is imbalanced, so the split is stratified and training uses class weights.
2. Preprocessing is fitted only on training data, reducing leakage risk.
3. ROC-AUC compares ranking quality; precision, recall and F1 show threshold behavior.
4. The same test rows and threshold are used for all three models.
5. Explainability is descriptive, not proof that a feature causes churn.

## Honest-use notes

- Do not claim a metric before running the training script.
- A model trained on this sample dataset should not be used for real customer decisions.
- Dataset categories and patterns may not represent another telecom business.
- The dashboard's single-customer prediction is a learning demonstration.

See `RESUME_PROJECT.md` for bullets to use after you run and understand the project.
