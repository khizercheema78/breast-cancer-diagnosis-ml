"""
Breast Cancer Diagnosis - Machine Learning Pipeline
---------------------------------------------------
Compares several classifiers on the Wisconsin Diagnostic Breast Cancer
dataset (bundled with scikit-learn), tunes the best one, and reports
clinically relevant metrics (recall on malignant cases matters most).

Run:  python train.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def load_data():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    # In this dataset 0 = malignant, 1 = benign. Flip so 1 = malignant (positive class).
    y = 1 - y
    return X, y


def eda(X: pd.DataFrame, y: pd.Series) -> None:
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Malignant: {y.sum()} ({y.mean():.1%}), Benign: {(1 - y).sum()}")
    corr = X.assign(malignant=y).corr()["malignant"].drop("malignant")
    top = corr.abs().sort_values(ascending=False).head(10)
    print("\nTop 10 features most correlated with malignancy:")
    print(corr[top.index].round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    corr[top.index].sort_values().plot.barh(ax=ax, color="#7aa2f7")
    ax.set_title("Top features correlated with malignancy")
    fig.tight_layout()
    fig.savefig(OUT / "feature_correlation.png", dpi=150)
    plt.close(fig)


def compare_models(X_train, y_train) -> pd.DataFrame:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "SVM (RBF)": SVC(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv,
                                scoring=["accuracy", "recall", "precision", "roc_auc"])
        rows.append({
            "model": name,
            "accuracy": scores["test_accuracy"].mean(),
            "recall": scores["test_recall"].mean(),
            "precision": scores["test_precision"].mean(),
            "roc_auc": scores["test_roc_auc"].mean(),
        })
    table = pd.DataFrame(rows).sort_values("recall", ascending=False)
    print("\n5-fold cross-validation (training set):")
    print(table.round(4).to_string(index=False))
    return table


def tune_and_evaluate(X_train, X_test, y_train, y_test) -> None:
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", LogisticRegression(max_iter=5000))])
    grid = GridSearchCV(
        pipe,
        {"model__C": [0.01, 0.1, 1, 10, 100], "model__class_weight": [None, "balanced"]},
        scoring="recall", cv=5,
    )
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"\nBest params: {grid.best_params_}")

    proba = best.predict_proba(X_test)[:, 1]
    pred = best.predict(X_test)
    print("\nHold-out test set results:")
    print(classification_report(y_test, pred, target_names=["benign", "malignant"], digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["benign", "malignant"],
                                            cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion matrix (test set)")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], name="Tuned Logistic Regression")
    axes[1].set_title("ROC curve (test set)")
    fig.tight_layout()
    fig.savefig(OUT / "evaluation.png", dpi=150)
    plt.close(fig)

    coefs = pd.Series(best.named_steps["model"].coef_[0], index=X_train.columns)
    print("\nMost influential features (standardised coefficients):")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8).round(3).to_string())


def main():
    X, y = load_data()
    eda(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    compare_models(X_train, y_train).to_csv(OUT / "model_comparison.csv", index=False)
    tune_and_evaluate(X_train, X_test, y_train, y_test)
    print(f"\nCharts and tables saved to ./{OUT}/")


if __name__ == "__main__":
    np.random.seed(RANDOM_STATE)
    main()
