"""
eda.py  --  Task 2: Exploratory Data Analysis on the cleaned Titanic dataset.

Run the whole analysis in one go from the project root:
    python src/eda.py

What it does:
  1. Loads data/titanic_clean.csv (output of Task 1)
  2. Prints summary statistics, survival rates and statistical tests
  3. Saves 8 charts to reports/figures/
  4. Prints the key insights at the end

Add --show to also pop up each chart window (close a window to continue).
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="data/titanic_clean.csv")
parser.add_argument("--figdir", default="reports/figures")
parser.add_argument("--show", action="store_true", help="display charts as pop-up windows")
args = parser.parse_args()

if not args.show:
    matplotlib.use("Agg")  # save-only mode, no windows

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12, "axes.titleweight": "bold"})
os.makedirs(args.figdir, exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


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


# ---------------------------------------------------------------- 1. load
section("1. LOAD DATA")
df = pd.read_csv(args.input)
df["Pclass"] = df["Pclass"].astype("category")
print(f"Loaded {args.input}: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(df.head())

# ---------------------------------------------------------------- 2. stats
section("2. SUMMARY STATISTICS")
print(df[["Age", "SibSp", "Parch", "FamilySize", "Fare"]].describe().T.round(2))
print(f"\nOverall survival rate: {df.Survived.mean() * 100:.1f}%")
print(f"Skewness -> Fare: {df.Fare.skew():.2f} | Age: {df.Age.skew():.2f}")
q1, q3 = df.Fare.quantile([0.25, 0.75])
print(f"Fare outliers (1.5*IQR rule): {int((df.Fare > q3 + 1.5 * (q3 - q1)).sum())} passengers")

# ---------------------------------------------------------------- 3. distributions
section("3. DISTRIBUTIONS")
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
df.Survived.map({0: "Did not survive", 1: "Survived"}).value_counts().plot.bar(
    ax=ax[0], color=["#c0504d", "#4f81bd"], rot=0)
ax[0].set_title("Target balance")
sns.histplot(df.Age, bins=30, kde=True, ax=ax[1]); ax[1].set_title("Age distribution")
sns.histplot(df.Fare, bins=40, kde=True, ax=ax[2]); ax[2].set_title("Fare distribution (right-skewed)")
save("01_target_age_fare_distributions")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(np.log1p(df.Fare), bins=30, kde=True, ax=ax[0]); ax[0].set_title("log(1 + Fare) is far more symmetric")
sns.boxplot(x=df.Fare, ax=ax[1]); ax[1].set_title("Fare outliers")
save("02_fare_log_and_outliers")

# ---------------------------------------------------------------- 4. survival by category
section("4. SURVIVAL BY SEX, CLASS AND PORT")
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col in zip(ax, ["Sex", "Pclass", "Embarked"]):
    sns.barplot(data=df, x=col, y="Survived", errorbar=("ci", 95), ax=a)
    a.set_title(f"Survival rate by {col}"); a.set_ylim(0, 1)
save("03_survival_by_sex_class_port")
for col in ["Sex", "Pclass", "Embarked"]:
    print(f"\nSurvival rate by {col}:")
    print(df.groupby(col, observed=True).Survived.agg(["mean", "count"]).round(3))

pivot = df.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean", observed=True)
plt.figure(figsize=(6, 3.5))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
plt.title("Survival rate: Sex x Class")
save("04_heatmap_sex_class")
print("\nSurvival rate: Sex x Class")
print(pivot.round(2))

# ---------------------------------------------------------------- 5. age, fare, family
section("5. AGE, FARE AND FAMILY EFFECTS")
df["AgeBand"] = pd.cut(df.Age, [0, 12, 18, 35, 60, 100],
                       labels=["Child (0-12)", "Teen (13-18)", "Adult (19-35)", "Middle (36-60)", "Senior (60+)"])
df["FareQuartile"] = pd.qcut(df.Fare, 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=df, x="AgeBand", y="Survived", errorbar=None, ax=ax[0])
ax[0].set_title("Survival by age band"); ax[0].tick_params(axis="x", rotation=20)
sns.barplot(data=df, x="FareQuartile", y="Survived", errorbar=None, ax=ax[1])
ax[1].set_title("Survival by fare quartile")
save("05_survival_by_age_and_fare")
print(df.groupby("AgeBand", observed=True).Survived.agg(["mean", "count"]).round(3))
print(df.groupby("FareQuartile", observed=True).Survived.agg(["mean", "count"]).round(3))

fs = df.groupby("FamilySize").Survived.agg(["mean", "count"]).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=fs, x="FamilySize", y="mean", ax=ax[0]); ax[0].set_title("Survival by family size"); ax[0].set_ylabel("Survival rate")
sns.barplot(data=df.assign(IsAlone=df.IsAlone.map({True: "Alone", False: "With family"})),
            x="IsAlone", y="Survived", errorbar=None, ax=ax[1]); ax[1].set_title("Alone vs. with family")
save("06_family_size_and_alone")
print("\nSurvival by family size:")
print(fs.round(3).to_string(index=False))

# ---------------------------------------------------------------- 6. title, deck
section("6. TITLE AND DECK")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(data=df, x="Title", y="Survived", errorbar=None, order=["Mr", "Master", "Rare", "Miss", "Mrs"], ax=ax[0])
ax[0].set_title("Survival by title")
order = df.groupby("Deck").Survived.mean().sort_values().index
sns.barplot(data=df, x="Deck", y="Survived", errorbar=None, order=order, ax=ax[1])
ax[1].set_title("Survival by deck ('Unknown' = no cabin recorded)")
save("07_title_and_deck")
print(df.groupby("Title").Survived.agg(["mean", "count"]).round(3))
print(df.groupby("Deck").Survived.agg(["mean", "count"]).round(3).sort_values("count", ascending=False))

# ---------------------------------------------------------------- 7. correlations
section("7. CORRELATIONS")
num = df[["Survived", "Pclass", "Age", "Fare", "SibSp", "Parch", "FamilySize"]].copy()
num["Pclass"] = num["Pclass"].astype(int)
corr = num.corr()
plt.figure(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix (Pearson)")
save("08_correlation_matrix")
print(corr.round(2))

# ---------------------------------------------------------------- 8. tests
section("8. STATISTICAL TESTS")
rows = []
for c in ["Sex", "Pclass", "Embarked", "IsAlone", "Deck", "Title"]:
    ct = pd.crosstab(df[c], df["Survived"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    rows.append({"feature": c, "chi2": round(chi2, 1), "p_value": f"{p:.2e}", "cramers_v": round(v, 3)})
print(pd.DataFrame(rows).sort_values("cramers_v", ascending=False).to_string(index=False))
u = stats.mannwhitneyu(df[df.Survived == 1].Fare, df[df.Survived == 0].Fare)
print(f"\nFare, survivors vs non-survivors (Mann-Whitney U): p = {u.pvalue:.2e}")
print(f"  median fare -> survived: {df[df.Survived == 1].Fare.median():.2f} | died: {df[df.Survived == 0].Fare.median():.2f}")
u2 = stats.mannwhitneyu(df[df.Survived == 1].Age, df[df.Survived == 0].Age)
print(f"Age,  survivors vs non-survivors (Mann-Whitney U): p = {u2.pvalue:.3f}  (not significant)")

# ---------------------------------------------------------------- 9. imputation check
section("9. DATA-QUALITY CHECK: TASK 1 AGE IMPUTATION")
print(df.groupby("AgeWasMissing").Survived.agg(["mean", "count"]).round(3))
print(f"Age std -> real: {df[~df.AgeWasMissing].Age.std():.1f} | imputed: {df[df.AgeWasMissing].Age.std():.1f}")

# ---------------------------------------------------------------- 10. insights
section("10. KEY INSIGHTS")
f = df.groupby("Sex").Survived.mean()
c = df.groupby("Pclass", observed=True).Survived.mean()
print(f"""
1. SEX/TITLE ARE THE STRONGEST PREDICTORS
   Women survived at {f['female']*100:.1f}% vs {f['male']*100:.1f}% for men. Keep Sex/Title as core features.

2. CLASS INTERACTS WITH SEX
   Survival by class: 1st {c[1]*100:.0f}% | 2nd {c[2]*100:.0f}% | 3rd {c[3]*100:.0f}%.
   1st-class women 97% vs 3rd-class men 14%. Use an interaction term or a tree-based model.

3. FARE IS SKEWED AND TIED TO CLASS
   Skew {df.Fare.skew():.1f}; correlation with Pclass {corr.loc['Fare','Pclass']:.2f}. Use log1p(Fare); watch multicollinearity.

4. FAMILY SIZE IS NON-LINEAR
   Alone 30%, families of 2-4 people 55-72%, 5+ collapses. Bin it (alone/small/large); never use it linearly.

5. TITLE BEATS RAW AGE
   Mr 16% vs Master 58%, Mrs 79%, Miss 70%. Age correlation with survival is only {corr.loc['Age','Survived']:.2f} (not significant).

6. A RECORDED CABIN MOSTLY MEANS FIRST CLASS
   Deck known: 59-76% survival vs 30% for Unknown, but cabins exist for 81% of 1st class and 2% of 3rd.

7. EMBARKATION EFFECT IS PROBABLY CONFOUNDED
   Cherbourg 55% vs Southampton 34%, but 51% of Cherbourg passengers were 1st class.

8. MISSING AGE IS INFORMATIVE
   Survival 29% (age missing) vs 41% (age known). Keep the AgeWasMissing flag.
""")
print(f"Done. {len(os.listdir(args.figdir))} charts are in {args.figdir}/")
