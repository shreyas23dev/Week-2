"""Train and compare supervised models on the local Titanic dataset.

The script creates classification and regression metric tables, a Markdown
report, a comparison chart, and a PDF report in this folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "titanic.csv"
CLASSIFICATION_CSV = BASE_DIR / "week2_classification_results.csv"
REGRESSION_CSV = BASE_DIR / "week2_regression_results.csv"
REPORT_MD = BASE_DIR / "week2_report.md"
REPORT_PDF = BASE_DIR / "week2_report.pdf"
CHART_FILE = BASE_DIR / "week2_model_comparison.png"
REPOSITORY_URL = "https://github.com/shreyas23dev/Week1"

NUMERIC_FEATURES = ["age", "sibsp", "parch"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLASSIFICATION_TARGET = "survived"
REGRESSION_TARGET = "fare"


NAVY = colors.HexColor("#183153")
BLUE = colors.HexColor("#2563EB")
PALE_BLUE = colors.HexColor("#EFF6FF")
DARK_GREY = colors.HexColor("#374151")


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_classification(data: pd.DataFrame) -> pd.DataFrame:
    X = data[FEATURES]
    y = data[CLASSIFICATION_TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Most-frequent baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }

    results: list[dict[str, float | str | int]] = []
    for name, estimator in models.items():
        model = Pipeline(
            steps=[("preprocessor", make_preprocessor()), ("model", estimator)]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "model": name,
                "test_rows": int(len(y_test)),
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "roc_auc": roc_auc_score(y_test, probability),
            }
        )

    output = pd.DataFrame(results).sort_values("f1", ascending=False)
    output.to_csv(CLASSIFICATION_CSV, index=False, float_format="%.4f")
    return output


def evaluate_regression(data: pd.DataFrame) -> pd.DataFrame:
    # Ticket fare is the numeric target; all predictors are available before
    # the outcome and do not include survival-related duplicate columns.
    regression_data = data.dropna(subset=[REGRESSION_TARGET]).copy()
    X = regression_data[FEATURES]
    y = regression_data[REGRESSION_TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Mean fare baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
    }

    results: list[dict[str, float | str | int]] = []
    for name, estimator in models.items():
        model = Pipeline(
            steps=[("preprocessor", make_preprocessor()), ("model", estimator)]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        results.append(
            {
                "model": name,
                "test_rows": int(len(y_test)),
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": rmse,
                "r2": r2_score(y_test, predictions),
            }
        )

    output = pd.DataFrame(results).sort_values("rmse")
    output.to_csv(REGRESSION_CSV, index=False, float_format="%.4f")
    return output


def create_comparison_chart(classification: pd.DataFrame, regression: pd.DataFrame) -> None:
    classification_plot = classification.sort_values("f1", ascending=True)
    regression_plot = regression.sort_values("rmse", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    fig.suptitle("Titanic Week 2 — Supervised Model Comparison", fontsize=15, weight="bold")

    axes[0].barh(classification_plot["model"], classification_plot["f1"], color="#2563EB")
    axes[0].set_title("Classification · F1 score")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("F1 (higher is better)")
    axes[0].grid(axis="x", alpha=0.2)

    axes[1].barh(regression_plot["model"], regression_plot["rmse"], color="#F97316")
    axes[1].set_title("Regression · RMSE")
    axes[1].set_xlabel("Fare units (lower is better)")
    axes[1].grid(axis="x", alpha=0.2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(CHART_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include="number").columns:
        if column != "test_rows":
            selected[column] = selected[column].map(lambda value: f"{value:.3f}")
    headers = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in selected.itertuples(index=False, name=None)]
    return "\n".join([headers, divider, *rows])


def create_markdown_report(
    data: pd.DataFrame,
    classification: pd.DataFrame,
    regression: pd.DataFrame,
) -> None:
    best_classifier = classification.iloc[0]
    best_regressor = regression.iloc[0]
    class_table = markdown_table(
        classification,
        ["model", "test_rows", "accuracy", "precision", "recall", "f1", "roc_auc"],
    )
    regression_table = markdown_table(regression, ["model", "test_rows", "mae", "rmse", "r2"])
    report = f"""# Week 2: Supervised Machine Learning Models

## Objective

This project trains and compares supervised classification and regression models with Scikit-learn using the Titanic passenger dataset. The classification task predicts whether a passenger survived. The regression task predicts ticket fare from passenger attributes. A fixed random seed and a held-out 20% test set make the comparisons repeatable.

## Dataset and preparation

The source data contains {len(data):,} passengers. Both tasks use age, number of siblings or spouses aboard (sibsp), number of parents or children aboard (parch), passenger class (pclass), sex, and embarkation port (embarked) as predictors. Missing numeric values are imputed with the median and numeric columns are standardized; missing categorical values are imputed with the most frequent category and categorical values are one-hot encoded. These transformations are fitted inside Scikit-learn pipelines on the training split only, then applied to the test split. This avoids using test-set information during preprocessing.

The classification target is survived (0 = did not survive, 1 = survived), and the split is stratified to preserve the target proportions. The regression target is fare. No target-derived columns such as alive were used as predictors.

## Classification comparison

Five classifiers were evaluated on the same held-out test rows: a most-frequent baseline, Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN). Accuracy is the overall fraction correct; precision and recall focus on the positive survival class; F1 is their harmonic mean; ROC AUC measures ranking across classification thresholds.

{class_table}

The top model by F1 on this particular split was **{best_classifier['model']}** (F1 = {best_classifier['f1']:.3f}, accuracy = {best_classifier['accuracy']:.3f}, ROC AUC = {best_classifier['roc_auc']:.3f}). Model rankings can change with a different split, data sample, or hyperparameter choice; the test metrics are not guarantees of real-world performance.

## Regression comparison

Linear Regression was evaluated against a mean-fare Dummy Regressor baseline using the same features and a held-out test split. MAE is the average absolute fare error, RMSE penalizes larger errors more strongly, and R² compares the model with predicting the test-set mean (higher is better; it can be negative).

{regression_table}

The lowest test RMSE in this comparison came from **{best_regressor['model']}** (RMSE = {best_regressor['rmse']:.3f}, MAE = {best_regressor['mae']:.3f}, R² = {best_regressor['r2']:.3f}). The comparison is a baseline experiment, not a tuned or production-ready fare estimator.

## Conclusion and limitations

This project demonstrates training, prediction, and evaluation for the requested supervised-learning algorithms. The classification models are compared with multiple complementary metrics, while the fare regression model is compared with a simple baseline. The Titanic data is historical and limited, and passenger-level patterns should not be interpreted causally. Only one holdout split was used; cross-validation and careful hyperparameter tuning would be appropriate follow-up work. Because fare is highly skewed and affected by ticket and family arrangements not fully represented here, its prediction errors may be substantial.

## Reproducibility

Run `python train_titanic_models.py` from the project folder. The script creates `week2_classification_results.csv`, `week2_regression_results.csv`, `week2_model_comparison.png`, this Markdown report, and `week2_report.pdf`.

Repository: {REPOSITORY_URL}
"""
    REPORT_MD.write_text(report, encoding="utf-8")


def create_pdf_report() -> None:
    markdown = REPORT_MD.read_text(encoding="utf-8")
    classification = pd.read_csv(CLASSIFICATION_CSV)
    regression = pd.read_csv(REGRESSION_CSV)

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=19 * mm,
        bottomMargin=19 * mm,
        title="Week 2: Supervised Machine Learning Models",
        author="AI Pioneers Internship Project",
    )
    sample = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleCustom", parent=sample["Title"], fontName="Helvetica-Bold",
            fontSize=22, leading=27, textColor=NAVY, alignment=TA_CENTER, spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom", parent=sample["Normal"], fontSize=10, leading=14,
            textColor=DARK_GREY, alignment=TA_CENTER, spaceAfter=16,
        ),
        "heading": ParagraphStyle(
            "HeadingCustom", parent=sample["Heading2"], fontName="Helvetica-Bold",
            fontSize=13, leading=16, textColor=NAVY, spaceBefore=10, spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCustom", parent=sample["BodyText"], fontName="Helvetica",
            fontSize=8.7, leading=12.2, textColor=DARK_GREY, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "CaptionCustom", parent=sample["Normal"], fontSize=8, leading=10,
            textColor=DARK_GREY, spaceAfter=5,
        ),
    }

    def make_table(frame: pd.DataFrame, number_columns: list[str]) -> Table:
        headers = list(frame.columns)
        rows = [headers]
        for _, record in frame.iterrows():
            formatted = []
            for column in headers:
                value = record[column]
                if column in number_columns:
                    formatted.append(f"{value:.3f}")
                elif column == "test_rows":
                    formatted.append(str(int(value)))
                else:
                    formatted.append(str(value))
            rows.append(formatted)
        col_width = (A4[0] - 40 * mm) / len(headers)
        table = Table(rows, colWidths=[col_width] * len(headers), repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("LEADING", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE_BLUE]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return table

    story = [
        Spacer(1, 4 * mm),
        Paragraph("WEEK 2", styles["title"]),
        Paragraph("Supervised Machine Learning Models", styles["subtitle"]),
    ]
    section_text = {
        "Objective": "This project trains and compares supervised classification and regression models with Scikit-learn using the Titanic passenger dataset. Classification predicts survival; regression predicts fare. A fixed random seed and held-out 20% test set make comparisons repeatable.",
        "Dataset and preparation": f"The source data contains {len(pd.read_csv(DATA_FILE)):,} passengers. Predictors are age, sibsp, parch, pclass, sex, and embarked. Missing numeric values are median-imputed and standardized; missing categorical values are mode-imputed and one-hot encoded. Preprocessing is fitted within each pipeline on training data only. The classification split is stratified, and no target-derived feature such as alive is used.",
        "Classification comparison": "A most-frequent baseline and four classifiers were evaluated on the same held-out test rows: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors. Accuracy is overall correctness; precision and recall describe positive-class predictions; F1 balances precision and recall; ROC AUC measures ranking across thresholds.",
        "Regression comparison": "Linear Regression is compared with a mean-fare Dummy Regressor baseline. MAE is average absolute error; RMSE penalizes large errors more strongly; R² compares the model against a mean prediction and can be negative.",
        "Conclusion and limitations": "The experiment demonstrates model training, prediction, and metric-based comparison. Results are specific to one holdout split, are not causal findings, and should not be treated as production performance. Cross-validation, tuning, and analysis of the skewed fare target are useful follow-up steps.",
    }

    story.extend([Paragraph("1. Objective", styles["heading"]), Paragraph(section_text["Objective"], styles["body"])])
    story.extend([Paragraph("2. Dataset and preparation", styles["heading"]), Paragraph(section_text["Dataset and preparation"], styles["body"])])
    story.extend([
        Paragraph("3. Classification results", styles["heading"]),
        Paragraph(section_text["Classification comparison"], styles["body"]),
        make_table(classification, ["accuracy", "precision", "recall", "f1", "roc_auc"]),
        Spacer(1, 3 * mm),
        Paragraph("4. Regression results", styles["heading"]),
        Paragraph(section_text["Regression comparison"], styles["body"]),
        make_table(regression, ["mae", "rmse", "r2"]),
        Spacer(1, 3 * mm),
        Image(str(CHART_FILE), width=168 * mm, height=69 * mm),
        Paragraph("Figure 1. F1 score and RMSE comparisons on held-out test sets; higher F1 and lower RMSE are preferred.", styles["caption"]),
        Paragraph("5. Conclusion and limitations", styles["heading"]),
        Paragraph(section_text["Conclusion and limitations"], styles["body"]),
        Paragraph(f"Repository: {REPOSITORY_URL}", styles["caption"]),
    ])

    def footer(canvas, document) -> None:
        canvas.saveState()
        page_width, _ = A4
        canvas.setStrokeColor(colors.HexColor("#D1D5DB"))
        canvas.line(20 * mm, 14 * mm, page_width - 20 * mm, 14 * mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(20 * mm, 9 * mm, "AI Pioneers Internship · Machine Learning")
        canvas.drawRightString(page_width - 20 * mm, 9 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Titanic dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required = set(FEATURES + [CLASSIFICATION_TARGET, REGRESSION_TARGET])
    absent = required.difference(data.columns)
    if absent:
        raise ValueError(f"Dataset is missing required columns: {sorted(absent)}")

    classification = evaluate_classification(data)
    regression = evaluate_regression(data)
    create_comparison_chart(classification, regression)
    create_markdown_report(data, classification, regression)
    create_pdf_report()

    print(f"Classification results:\n{classification.to_string(index=False)}")
    print(f"\nRegression results:\n{regression.to_string(index=False)}")
    print(f"\nSaved report: {REPORT_MD}")
    print(f"Saved PDF: {REPORT_PDF}")
    print(f"Saved comparison chart: {CHART_FILE}")


if __name__ == "__main__":
    main()
