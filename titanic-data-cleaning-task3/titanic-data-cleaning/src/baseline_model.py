"""
baseline_model.py  --  Task 3: baseline classification model on the cleaned Titanic dataset.

Run from the project root:
    python src/baseline_model.py

What it does:
  1. Loads data/titanic_clean.csv (output of Task 1)
  2. Trains a naive majority-class baseline and a Logistic Regression baseline
  3. Evaluates both with accuracy, precision, recall, F1, ROC-AUC
  4. Runs 5-fold cross-validation
  5. Prints feature importance and saves a confusion-matrix/ROC figure and a
     feature-importance figure to reports/figures/
  6. Prints a limitations summary

Add --show to also pop up each chart window.
"""

import argparse
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/titanic_clean.csv")
parser.add_argument("--figdir", default="reports/figures")
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                              confusion_matrix, ConfusionMatrixDisplay, f1_score,
                              precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_validate,  # noqa: E402
                                      train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12, "axes.titleweight": "bold"})
os.makedirs(args.figdir, exist_ok=True)
pd.set_option("display.width", 140)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save(name):
    plt.tight_layout()
    path = os.path.join(args.figdir, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [chart saved] {path}")
    if args.show:
        plt.show()
    plt.close()


section("1. LOAD DATA")
df = pd.read_csv(args.input)
print(f"Loaded {args.input}: {df.shape[0]} rows x {df.shape[1]} columns")

target = "Survived"
num_feats = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
cat_feats = ["Pclass", "Sex", "Embarked", "Title", "Deck", "IsAlone", "AgeWasMissing"]
X = df[num_feats + cat_feats]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
print(f"Survival rate -> train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_feats),
])

section("2. NAIVE BASELINE")
dummy = Pipeline([("pre", preprocess), ("clf", DummyClassifier(strategy="most_frequent"))])
dummy.fit(X_train, y_train)
dummy_acc = accuracy_score(y_test, dummy.predict(X_test))
print(f"Always predict 'did not survive' -> accuracy: {dummy_acc:.3f}")

section("3. LOGISTIC REGRESSION BASELINE")
logreg = Pipeline([("pre", preprocess), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_proba = logreg.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1:        {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}")
print(f"Improvement over naive baseline: +{acc - dummy_acc:.3f} accuracy")
print("\n" + classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"], digits=3))

section("4. CONFUSION MATRIX & ROC CURVE")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Did not survive", "Survived"]).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Confusion matrix (test set)")
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax[1].plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})", linewidth=2)
ax[1].plot([0, 1], [0, 1], "--", color="grey", label="Random guess (AUC = 0.50)")
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title("ROC curve"); ax[1].legend(loc="lower right")
save("09_confusion_matrix_and_roc")

section("5. CROSS-VALIDATION (5-FOLD)")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_validate(logreg, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    scores = cv_scores[f"test_{metric}"]
    print(f"{metric:10s} mean={scores.mean():.3f}  std={scores.std():.3f}  folds={np.round(scores, 3)}")

section("6. FEATURE IMPORTANCE")
feat_names = logreg.named_steps["pre"].get_feature_names_out()
coefs = logreg.named_steps["clf"].coef_[0]
importance = pd.Series(coefs, index=feat_names).sort_values(key=np.abs, ascending=False)
print(importance.head(12).round(3))

plt.figure(figsize=(8, 6))
top = importance.head(12)
colors = ["#4f81bd" if v > 0 else "#c0504d" for v in top.values]
sns.barplot(x=top.values, y=top.index, palette=colors, hue=top.index, legend=False)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Top 12 features by |coefficient| (blue = pushes toward survival)")
plt.xlabel("Logistic regression coefficient (standardized inputs)")
save("10_feature_importance")

section("7. LIMITATIONS")
print("""
1. Sex and Title are collinear (both encode almost the same information).
2. Some Deck categories have only 1-4 passengers -> unstable estimates.
3. 891 rows is small; cross-validation shows real fold-to-fold variance.
4. ~20% of Age values are Task 1's imputed estimates, not observed data.
5. No hyperparameter tuning or feature selection -- this is a baseline.
6. The default 0.5 decision threshold was not tuned for precision/recall trade-offs.
7. Fairness concern: Sex, Pclass and Deck (a class proxy) are the strongest
   predictors, reflecting historical inequities ("women and children first",
   unequal lifeboat access by class). Suitable only as a historical/educational
   exercise, not as a template for real triage or resource-allocation decisions.
8. Results are for a single random seed (random_state=42).
""")
print(f"Done. Charts saved in {args.figdir}/")
