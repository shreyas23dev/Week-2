# Week 2: Supervised Machine Learning Models

## Objective

This project trains and compares supervised classification and regression models with Scikit-learn using the Titanic passenger dataset. The classification task predicts whether a passenger survived. The regression task predicts ticket fare from passenger attributes. A fixed random seed and a held-out 20% test set make the comparisons repeatable.

## Dataset and preparation

The source data contains 891 passengers. Both tasks use age, number of siblings or spouses aboard (sibsp), number of parents or children aboard (parch), passenger class (pclass), sex, and embarkation port (embarked) as predictors. Missing numeric values are imputed with the median and numeric columns are standardized; missing categorical values are imputed with the most frequent category and categorical values are one-hot encoded. These transformations are fitted inside Scikit-learn pipelines on the training split only, then applied to the test split. This avoids using test-set information during preprocessing.

The classification target is survived (0 = did not survive, 1 = survived), and the split is stratified to preserve the target proportions. The regression target is fare. No target-derived columns such as alive were used as predictors.

## Classification comparison

Five classifiers were evaluated on the same held-out test rows: a most-frequent baseline, Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN). Accuracy is the overall fraction correct; precision and recall focus on the positive survival class; F1 is their harmonic mean; ROC AUC measures ranking across classification thresholds.

| model | test_rows | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 179 | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision Tree | 179 | 0.793 | 0.833 | 0.580 | 0.684 | 0.820 |
| K-Nearest Neighbors | 179 | 0.788 | 0.816 | 0.580 | 0.678 | 0.834 |
| Random Forest | 179 | 0.771 | 0.750 | 0.609 | 0.672 | 0.850 |
| Most-frequent baseline | 179 | 0.615 | 0.000 | 0.000 | 0.000 | 0.500 |

The top model by F1 on this particular split was **Logistic Regression** (F1 = 0.724, accuracy = 0.804, ROC AUC = 0.844). Model rankings can change with a different split, data sample, or hyperparameter choice; the test metrics are not guarantees of real-world performance.

## Regression comparison

Linear Regression was evaluated against a mean-fare Dummy Regressor baseline using the same features and a held-out test split. MAE is the average absolute fare error, RMSE penalizes larger errors more strongly, and R² compares the model with predicting the test-set mean (higher is better; it can be negative).

| model | test_rows | mae | rmse | r2 |
| --- | --- | --- | --- | --- |
| Linear Regression | 179 | 18.870 | 30.920 | 0.382 |
| Mean fare baseline | 179 | 25.692 | 39.383 | -0.002 |

The lowest test RMSE in this comparison came from **Linear Regression** (RMSE = 30.920, MAE = 18.870, R² = 0.382). The comparison is a baseline experiment, not a tuned or production-ready fare estimator.

## Conclusion and limitations

This project demonstrates training, prediction, and evaluation for the requested supervised-learning algorithms. The classification models are compared with multiple complementary metrics, while the fare regression model is compared with a simple baseline. The Titanic data is historical and limited, and passenger-level patterns should not be interpreted causally. Only one holdout split was used; cross-validation and careful hyperparameter tuning would be appropriate follow-up work. Because fare is highly skewed and affected by ticket and family arrangements not fully represented here, its prediction errors may be substantial.

## Reproducibility

Run `python train_titanic_models.py` from the project folder. The script creates `week2_classification_results.csv`, `week2_regression_results.csv`, `week2_model_comparison.png`, this Markdown report, and `week2_report.pdf`.

Repository: https://github.com/shreyas23dev/Week1
