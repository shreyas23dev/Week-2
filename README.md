# Week 2 — Supervised Machine Learning Models

AI Pioneers Internship (Machine Learning Internship).

This project trains and compares supervised classification and regression models on the Titanic passenger dataset with Scikit-learn.

## Tasks

- **Classification:** Predict passenger survival using Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN), compared with a most-frequent baseline.
- **Regression:** Predict ticket fare using Linear Regression, compared with a mean-fare baseline.
- **Evaluation:** Classification metrics include accuracy, precision, recall, F1, and ROC AUC. Regression metrics include MAE, RMSE, and R².

Preprocessing is performed in Scikit-learn pipelines: numeric missing values are median-imputed and standardized; categorical values are mode-imputed and one-hot encoded. The pipelines are fitted only on training partitions. A deterministic 80/20 train/test split is used; the classification split is stratified.

## Run

Install dependencies if needed:

```powershell
python -m pip install pandas numpy matplotlib reportlab scikit-learn
```

Place `titanic.csv` beside the script, then run:

```powershell
python train_titanic_models.py
```

The script generates classification and regression result CSVs, a comparison chart, and Markdown and PDF reports.

## Files

- `train_titanic_models.py` — model training, evaluation, chart, and report generator.
- `titanic.csv` — source dataset used for the experiment.
- `week2_classification_results.csv` — classification metrics.
- `week2_regression_results.csv` — regression metrics.
- `week2_model_comparison.png` — chart comparing F1 and RMSE.
- `week2_report.md` and `week2_report.pdf` — report outputs.

## Dataset

Seaborn Titanic example dataset: https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv

The findings are specific to the provided data and one test split; they should not be interpreted as causal or guaranteed real-world model performance.
