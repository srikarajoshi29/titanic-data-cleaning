# Titanic Dataset — Data Cleaning

A small, self-contained project that takes the public **Titanic passenger
manifest** dataset, cleans missing values, fixes column types, and documents
every cleaning decision made along the way.

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
│   └── titanic_cleaning.ipynb # exploratory + cleaning walkthrough, with outputs
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

# Option B: open the notebook
jupyter notebook notebooks/titanic_cleaning.ipynb
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
