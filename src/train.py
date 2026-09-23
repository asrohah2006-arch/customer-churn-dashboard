"""Train and compare Logistic Regression, Random Forest, and a Keras ANN."""
import argparse, json, os, random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from data import load_and_clean, split_features_target, build_preprocessor
from evaluate import save_evaluation

SEED = 42

def set_seed():
    os.environ["PYTHONHASHSEED"] = str(SEED); random.seed(SEED); np.random.seed(SEED)

def main(data_path, model_dir, report_dir, epochs):
    set_seed()
    model_dir, report_dir = Path(model_dir), Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(data_path); X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)
    split_note = {"seed": SEED, "test_size": 0.20, "train_rows": len(X_train), "test_rows": len(X_test), "positive_rate_train": float(y_train.mean()), "positive_rate_test": float(y_test.mean())}
    (report_dir / "split.json").write_text(json.dumps(split_note, indent=2))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=SEED),
    }
    results = {}
    fitted = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", build_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        results[name] = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        joblib.dump(pipe, model_dir / ("logistic_regression.joblib" if name.startswith("Logistic") else "random_forest.joblib"))

    # Bounded, model-agnostic fallback explainability on at most 1,000 held-out rows.
    sample_n = min(1000, len(X_test)); X_explain = X_test.sample(sample_n, random_state=SEED); y_explain = y_test.loc[X_explain.index]
    perm = permutation_importance(fitted["Random Forest"], X_explain, y_explain, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
    pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False).to_csv(report_dir / "permutation_importance.csv", index=False)

    # Optional SHAP. Failure does not stop the project because permutation importance is always produced.
    try:
        import shap
        rf_pipe = fitted["Random Forest"]; prep = rf_pipe.named_steps["preprocess"]; model = rf_pipe.named_steps["model"]
        transformed = prep.transform(X_explain.iloc[:250]); names = prep.get_feature_names_out()
        sv = shap.TreeExplainer(model).shap_values(transformed)
        values = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 0) == 3 else sv)
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).to_csv(report_dir / "shap_importance.csv", index=False)
    except Exception as exc:
        (report_dir / "shap_status.txt").write_text(f"SHAP was skipped; permutation importance is available. Reason: {type(exc).__name__}: {exc}\n")

    # TensorFlow is imported here so data/EDA and ML tests can run without it.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is required for the ANN. Install requirements.txt, then rerun train.py.") from exc
    tf.keras.utils.set_random_seed(SEED)
    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train).astype("float32"); X_test_t = preprocessor.transform(X_test).astype("float32")
    joblib.dump(preprocessor, model_dir / "ann_preprocessor.joblib")
    ann = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_t.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"), tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    classes = np.array([0, 1]); weights = compute_class_weight("balanced", classes=classes, y=y_train)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)]
    history = ann.fit(X_train_t, y_train, validation_split=0.20, epochs=epochs, batch_size=64, class_weight=dict(zip(classes, weights)), callbacks=callbacks, verbose=1)
    ann.save(model_dir / "ann.keras")
    pd.DataFrame(history.history).to_csv(report_dir / "ann_history.csv", index=False)
    results["Neural Network"] = ann.predict(X_test_t, verbose=0).ravel()
    metrics = save_evaluation(results, y_test, report_dir)
    print(metrics.to_string(index=False))
    print("Saved metrics to", report_dir / "metrics.csv")
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data", default="data/raw/Telco-Customer-Churn.csv"); p.add_argument("--models", default="models"); p.add_argument("--reports", default="reports"); p.add_argument("--epochs", type=int, default=80)
    a=p.parse_args(); main(a.data, a.models, a.reports, a.epochs)
