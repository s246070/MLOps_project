import os
import pandas as pd

def make_reference_csv(
    raw_path: str = "data/raw/Titanic-Dataset.csv",
    out_path: str = "data/reference/titanic_features_reference.csv",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_csv(raw_path)

    
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")

    
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # remove the target to make a new csv-file without the target
    if "Survived" in df.columns:
        df = df.drop(columns=["Survived"])

    
    df.to_csv(out_path, index=False)
    print(f"Saved reference features to {out_path}")
