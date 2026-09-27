"""
clean_titanic.py

Cleans the raw Titanic passenger dataset (Kaggle / data.world "Titanic" data,
mirrored at https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).

Run directly:
    python src/clean_titanic.py --input data/titanic_raw.csv --output data/titanic_clean.csv

Or import:
    from src.clean_titanic import clean_titanic
    df_clean = clean_titanic(df_raw)

All cleaning decisions and assumptions are documented inline and in README.md.
"""

import argparse
import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    return pd.read_csv(path)


def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Titanic dataframe.

    Steps:
      1. Drop exact duplicate rows.
      2. Fix / enforce column dtypes.
      3. Handle missing values column by column.
      4. Engineer a couple of small helper columns used downstream.
      5. Drop columns that are not usable for analysis given how sparse
         or high-cardinality they are (documented below).

    Returns a new, cleaned DataFrame. The input is not modified in place.
    """
    df = df.copy()

    # --- 1. Duplicates -----------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} exact duplicate row(s).")

    # --- 2. Type fixes -------------------------------------------------------
    # PassengerId is a row identifier, not a numeric feature -> keep as int
    # but never use it in modeling/aggregation.
    df["PassengerId"] = df["PassengerId"].astype(int)

    # Survived and Pclass are categorical codes stored as integers.
    df["Survived"] = df["Survived"].astype("Int64")   # nullable int, 0/1
    df["Pclass"] = df["Pclass"].astype("category")

    # Sex / Embarked are categorical strings.
    df["Sex"] = df["Sex"].astype("category")

    # SibSp / Parch are counts -> keep as integers.
    df["SibSp"] = df["SibSp"].astype(int)
    df["Parch"] = df["Parch"].astype(int)

    # Fare and Age are continuous -> ensure float.
    df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    # --- 3. Missing values ---------------------------------------------------

    # Age (177 missing, ~20% of rows):
    # ASSUMPTION: age correlates with passenger title (Mr/Mrs/Miss/Master) and
    # class, so instead of a single global median we impute using the median
    # age within each (Title, Pclass) group. This is a documented, defensible
    # choice; a global median would flatten known age differences between,
    # e.g., "Master" (young boys) and "Mr" (adult men).
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")[0].str.strip()
    # Collapse rare titles into a small set of buckets.
    rare_titles = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
        "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
        "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
    }
    df["Title"] = df["Title"].replace(rare_titles)

    df["AgeWasMissing"] = df["Age"].isna()  # flag captured BEFORE filling, for transparency
    age_by_group = df.groupby(["Title", "Pclass"], observed=True)["Age"].transform("median")
    overall_median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(age_by_group)
    # Fallback for any (Title, Pclass) combo that itself had no known ages.
    df["Age"] = df["Age"].fillna(overall_median_age)

    # Cabin (687 missing, ~77% of rows):
    # ASSUMPTION: this column is too sparse to reliably impute a specific cabin
    # number. Instead of guessing a cabin, we extract the deck letter (first
    # character of the cabin code) where known, and explicitly label the rest
    # as "Unknown" rather than dropping the column entirely, since "cabin
    # known vs. unknown" itself is a meaningful, complete signal (it strongly
    # correlates with class and survival in this dataset).
    df["Deck"] = df["Cabin"].str[0]
    df["Deck"] = df["Deck"].fillna("Unknown")
    df = df.drop(columns=["Cabin"])

    # Embarked (2 missing):
    # ASSUMPTION: with only 2 missing values out of 891, we impute using the
    # mode (most frequent port, "S" = Southampton) rather than dropping the
    # rows, since the information loss from dropping is unnecessary and the
    # mode is a low-risk fill for such a small number of cases.
    embarked_mode = df["Embarked"].mode(dropna=True)[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    df["Embarked"] = df["Embarked"].astype("category")

    # Fare: no missing values in this dataset, but guard defensively in case
    # a different snapshot has some -> impute with median fare for that Pclass.
    if df["Fare"].isna().any():
        fare_by_class = df.groupby("Pclass", observed=True)["Fare"].transform("median")
        df["Fare"] = df["Fare"].fillna(fare_by_class)

    # --- 4. Small helper / engineered columns --------------------------------
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1)

    # --- 5. Columns dropped and why ------------------------------------------
    # - "Name": high-cardinality free text, information of value (title) already
    #   extracted into "Title"; the raw name is not useful for tabular analysis.
    # - "Ticket": alphanumeric ticket codes with no consistent structure and
    #   very high cardinality (681 unique values / 891 rows); not usable as-is.
    df = df.drop(columns=["Name", "Ticket"])

    # Final column order for readability.
    ordered_cols = [
        "PassengerId", "Survived", "Pclass", "Title", "Sex", "Age",
        "AgeWasMissing", "SibSp", "Parch", "FamilySize", "IsAlone", "Fare",
        "Deck", "Embarked",
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]

    return df


def main():
    parser = argparse.ArgumentParser(description="Clean the raw Titanic dataset.")
    parser.add_argument("--input", default="data/titanic_raw.csv", help="Path to raw CSV.")
    parser.add_argument("--output", default="data/titanic_clean.csv", help="Path to write cleaned CSV.")
    args = parser.parse_args()

    df_raw = load_data(args.input)
    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")
    print("Missing values per column (raw):")
    print(df_raw.isna().sum()[df_raw.isna().sum() > 0])

    df_clean = clean_titanic(df_raw)
    print(f"\nCleaned data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")
    print("Missing values per column (clean):")
    missing_clean = df_clean.isna().sum()
    print(missing_clean[missing_clean > 0] if missing_clean.sum() else "None")

    df_clean.to_csv(args.output, index=False)
    print(f"\nSaved cleaned dataset to: {args.output}")


if __name__ == "__main__":
    main()
