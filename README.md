# 🩺 Breast Cancer Diagnosis with Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-complete-success)

An end-to-end machine learning pipeline that classifies breast tumours as **malignant** or **benign** from 30 cell-nucleus measurements (Wisconsin Diagnostic Breast Cancer dataset).

In medical screening, **missing a malignant case is far worse than a false alarm**, so models are selected and tuned for **recall on the malignant class**, not just accuracy.

## 🔍 What the project does

1. **Exploratory analysis:** class balance and the features most correlated with malignancy
2. **Model comparison:** Logistic Regression, SVM, Random Forest and Gradient Boosting with 5-fold stratified cross-validation
3. **Hyperparameter tuning:** `GridSearchCV` that optimises recall, including class weighting
4. **Evaluation on a held-out test set:** classification report, ROC-AUC, confusion matrix and ROC curve
5. **Interpretability:** standardised coefficients show which measurements drive the prediction

## 📈 Results

**Cross-validation (training set, 5 folds)**

| Model | Accuracy | Recall (malignant) | Precision | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | **0.974** | **0.953** | 0.977 | **0.996** |
| SVM (RBF) | 0.971 | 0.947 | 0.977 | 0.995 |
| Gradient Boosting | 0.967 | 0.947 | 0.965 | 0.991 |
| Random Forest | 0.960 | 0.935 | 0.958 | 0.988 |

**Tuned Logistic Regression on the unseen test set (114 patients)**

| Metric | Score |
|---|---|
| Accuracy | **99.1%** |
| Recall (malignant) | **97.6%** |
| Precision (malignant) | **100%** |
| ROC-AUC | **0.998** |

**Key insights**

- Shape and size features (*worst concave points*, *worst perimeter*, *worst radius*) are the strongest signals of malignancy.
- A simple, well-regularised linear model beats complex ensembles here. It is also easier to explain to clinicians.
- `class_weight="balanced"` improves sensitivity to the minority (malignant) class.

## 🚀 Run it

```bash
git clone https://github.com/khizercheema78/breast-cancer-diagnosis-ml.git
cd breast-cancer-diagnosis-ml
pip install -r requirements.txt
python train.py
```

Charts (`feature_correlation.png`, `evaluation.png`) and `model_comparison.csv` are saved in `outputs/`.

## 🧰 Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib

## 📚 Dataset

[Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), UCI Machine Learning Repository, bundled with scikit-learn. 569 samples, 30 features.

> ⚠️ Educational project. Not a medical device and not for clinical use.
