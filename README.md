# Titanic Dataset — Data Cleaning & EDA

A two-part project on the public **Titanic passenger manifest** dataset:

- **Task 1 — Data cleaning:** clean missing values, fix column types, document every assumption (`notebooks/titanic_cleaning.ipynb`).
- **Task 2 — Exploratory Data Analysis:** statistics, visualizations and modelling/decision insights on the cleaned data (`notebooks/02_titanic_eda.ipynb`).

## Dataset

- **Name:** Titanic passenger data (the classic "Titanic: Machine Learning
  from Disaster" dataset).
- **Source:** Public mirror on GitHub —
  [`datasciencedojo/datasets/titanic.csv`](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
  (this is the same dataset used for the well-known Kaggle Titanic
  competition training set).
- **Size:** 891 rows × 12 columns.
- **License:** publicly available for educational/analytical use; no
  personally sensitive data (this is a century-old historical passenger
  list).

A local copy of the raw file is included at `data/titanic_raw.csv` so the
project is fully reproducible without needing internet access.

## What's in this repo

```
.
├── data/
│   ├── titanic_raw.csv        # original, unmodified dataset
│   └── titanic_clean.csv      # output of the cleaning process
├── notebooks/
│   ├── titanic_cleaning.ipynb # Task 1: cleaning walkthrough, with outputs
│   └── 02_titanic_eda.ipynb   # Task 2: EDA, statistics, visualizations, insights
├── reports/figures/           # 8 PNG charts exported from the EDA notebook
├── src/
│   └── clean_titanic.py       # reusable cleaning function + CLI script
├── README.md
└── requirements.txt
```

## How to run it

```bash
pip install -r requirements.txt

# Option A: run the script directly
python src/clean_titanic.py --input data/titanic_raw.csv --output data/titanic_clean.csv

# Option B: open the notebooks (run Task 1 first, EDA reads its output)
jupyter notebook notebooks/titanic_cleaning.ipynb
jupyter notebook notebooks/02_titanic_eda.ipynb
```

## Missing values found (raw data)

| Column     | Missing count | Missing % |
|------------|---------------|-----------|
| `Age`      | 177           | 19.9%     |
| `Cabin`    | 687           | 77.1%     |
| `Embarked` | 2             | 0.2%      |

No other columns had missing values. After cleaning, **0** missing values
remain in `data/titanic_clean.csv`.

## Cleaning decisions and assumptions

Every decision below is also inline-commented in `src/clean_titanic.py`.

| Column | Issue | Decision | Assumption / rationale |
|---|---|---|---|
| `Age` | 177 missing (~20%) | Impute with the **median age within each (Title, Pclass) group**, where `Title` (Mr/Mrs/Miss/Master/Rare) is extracted from the `Name` field | Age is closely tied to title and passenger class (e.g. "Master" = young boy, "Mrs" = adult woman); a single global median would erase that structure. A boolean `AgeWasMissing` flag is kept so imputed rows remain identifiable. |
| `Cabin` | 687 missing (~77%) | Extract the **deck letter** (first character of the cabin code) into a new `Deck` column; missing → `"Unknown"`. Original `Cabin` column dropped. | The column is too sparse to impute a specific cabin number, and doing so would fabricate information. Whether a cabin was recorded at all is itself meaningful (it correlates with ticket class/fare), so that signal is preserved as an explicit `"Unknown"` category rather than silently dropped. |
| `Embarked` | 2 missing (~0.2%) | Impute with the **mode** (`"S"`, Southampton) | Only 2 of 891 rows are affected; mode imputation is low-risk here and preserves otherwise-complete records rather than dropping them. |
| `Fare` | 0 missing in this snapshot, but defensively handled | If any were missing: median fare **within the same `Pclass`** | Fare varies a lot by class, so a per-class median is more accurate than a single global fare. Included defensively since other releases of this dataset do have a missing fare. |
| `Name` | High-cardinality free text (891 unique values) | Extract `Title`, then **drop** the raw name | The raw name string isn't a usable analytical feature; the socially meaningful part (title) is kept. |
| `Ticket` | High-cardinality, inconsistent alphanumeric codes (681 unique values) | **Dropped** | No consistent structure to parse and too high-cardinality to use as a categorical feature. |
| `Pclass`, `Sex`, `Embarked` | Stored as generic numeric/string types | Cast to pandas **`category`** dtype | These are genuinely categorical variables (3, 2, and 3 levels respectively), not continuous numbers or free text. |
| `Survived` | Stored as plain `int64` | Cast to nullable **`Int64`** | Defensive choice: keeps the column able to hold missing labels without breaking dtype, matching how the original Kaggle test split (labels withheld) would need to be handled. |
| Exact duplicate rows | None found in this dataset | Dropped if present | Standard defensive step; `drop_duplicates()` is a no-op here but guards against future re-runs on updated data pulls. |

### Engineered columns

- `FamilySize` = `SibSp` + `Parch` + 1 (self included).
- `IsAlone` = `True` when `FamilySize == 1`.
- `Title` — extracted from `Name`, with rare titles (`Lady`, `Countess`,
  `Capt`, `Col`, `Don`, `Dr`, `Major`, `Rev`, `Sir`, `Jonkheer`, `Dona`)
  collapsed into `"Rare"`, and `Mlle`/`Ms` → `"Miss"`, `Mme` → `"Mrs"`.
- `AgeWasMissing` — `True` for rows where `Age` was imputed.
- `Deck` — first character of `Cabin`, or `"Unknown"` if `Cabin` was missing.

## Output schema (`data/titanic_clean.csv`)

| Column | Type | Notes |
|---|---|---|
| `PassengerId` | int | Row identifier only — not a feature |
| `Survived` | Int64 | 0 = did not survive, 1 = survived |
| `Pclass` | category | 1st / 2nd / 3rd class |
| `Title` | category | Extracted from name |
| `Sex` | category | male / female |
| `Age` | float | Cleaned, no missing values |
| `AgeWasMissing` | bool | True if `Age` was imputed |
| `SibSp` | int | # siblings/spouses aboard |
| `Parch` | int | # parents/children aboard |
| `FamilySize` | int | Engineered: `SibSp + Parch + 1` |
| `IsAlone` | bool | Engineered: `FamilySize == 1` |
| `Fare` | float | Ticket fare |
| `Deck` | category | First letter of cabin, or "Unknown" |
| `Embarked` | category | Port of embarkation (C/Q/S) |

## Requirements

See `requirements.txt`. Core dependency is `pandas`; `jupyter`/`nbconvert`
are only needed to run/re-execute the notebook.

---

## Task 2 — EDA summary

**Approach:** summary statistics and skewness checks; target balance; distributions (with a log transform for `Fare`); survival rates by category with confidence intervals; a Sex x Class heatmap; correlation matrix; chi-square (with Cramér's V) and Mann-Whitney U tests; and a data-quality check of Task 1's Age imputation.

**Headline numbers:** overall survival 38.4%; women 74.2% vs men 18.9%; 1st/2nd/3rd class 63% / 47% / 24%.

### Key insights (details and charts in the notebook)

| # | Insight | Recommended modelling / decision action |
|---|---|---|
| 1 | Sex (and Title) are the strongest predictors (Cramér's V ≈ 0.54 / 0.57) | Core features; never drop |
| 2 | Class effect interacts with sex: 1st-class women 97% vs 3rd-class men 14% | Interaction term or tree-based model |
| 3 | `Fare` is heavily right-skewed (skew ≈ 4.8) and correlated with class (-0.55) | `log1p(Fare)`, watch multicollinearity |
| 4 | Family size is non-linear: alone 30%, 2-4 people 55-72%, 5+ collapses | Bin into alone / small / large; do not use linearly |
| 5 | Title carries more signal than raw Age (Age vs. survival r = -0.06, not significant) | Use Title and age bands |
| 6 | Known cabin/deck is largely a proxy for 1st class (81% of 1st class vs 2% of 3rd) | Keep the "Unknown" category; interpret with care |
| 7 | Cherbourg's higher survival (55%) is probably confounded by class (51% 1st class) | Validate against class before using |
| 8 | Missing-age rows survive less (29% vs 41%) and have lower variance after imputation | Keep the `AgeWasMissing` flag |

### Note: fix made to Task 1
While preparing the EDA, the title `the Countess` was found to be missing from the rare-title grouping in `src/clean_titanic.py`. It is now mapped to `Rare` and `data/titanic_clean.csv` was regenerated (`Title` now has 5 clean categories).

### Limitations
Observational, historical data (891 of ~2,200 people aboard): patterns are correlations within this sample, not causal claims. Some groups (decks A/F/G, family size 7+) are small, so their rates are noisy.
