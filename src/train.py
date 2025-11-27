
import os
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import CONFIG
from data_loader import load_table
from preprocess import build_preprocessor, split_X_y
from models import build_models


def ensure_dirs(base_dir: str) -> Dict[str, Path]:
    base = Path(base_dir)
    models_dir = base / "models"
    plots_dir = base / "plots"
    reports_dir = base / "reports"

    for d in (models_dir, plots_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "models": models_dir,
        "plots": plots_dir,
        "reports": reports_dir,
    }


def train_and_evaluate() -> None:
    paths = ensure_dirs(CONFIG.results_dir)

    # 1) Load data
    df = load_table(CONFIG.db_path)
    print(f"Loaded data with shape: {df.shape}")

    # 2) Split X, y
    X, y = split_X_y(df)
    print(f"Features shape: {X.shape}, target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG.test_size,
        random_state=CONFIG.random_state,
        stratify=y,
    )

    # 3) Preprocessor + models
    preprocessor = build_preprocessor()
    model_dict = build_models()

    summary_lines = []

    for name, clf in model_dict.items():
        print(f"\n=== Training model: {name} ===")

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf),
            ]
        )

        pipe.fit(X_train, y_train)

        # predictions & probabilities
        y_pred = pipe.predict(X_test)
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            # fallback: use decision_function if available
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_test)
                # convert to pseudo-prob with logistic function
                y_proba = 1 / (1 + np.exp(-scores))
            else:
                y_proba = None

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")

        if y_proba is not None:
            auc = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC: {auc:.4f}")
        else:
            auc = None
            print("ROC AUC: not available (no probabilities)")

        print("Classification report:")
        report = classification_report(y_test, y_pred, digits=4)
        print(report)

        # Save model
        model_path = paths["models"] / f"{name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved model to: {model_path}")

        # Save ROC curve
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend()
            plot_path = paths["plots"] / f"roc_{name}.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print(f"Saved ROC plot to: {plot_path}")

        # Save text report
        report_path = paths["reports"] / f"{name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"ROC AUC: {auc}\n\n")
            f.write(report)
        print(f"Saved classification report to: {report_path}")

        summary_lines.append(
            f"{name},accuracy={acc:.4f},roc_auc={auc}\n"
        )

    # Global summary
    summary_path = paths["base"] / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("model,accuracy,roc_auc\n")
        for line in summary_lines:
            f.write(line)

    print(f"\nTraining complete. Summary written to: {summary_path}")


if __name__ == "__main__":
    train_and_evaluate()
