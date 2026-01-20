import os
import pickle
import torch
import pandas as pd
import typer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def preprocess_data(
    raw_path: str = "data/raw/Titanic-Dataset.csv",
    processed_dir: str = "data/processed",
) -> None:
    """Preprocess Titanic CSV and save as pickle files."""
    os.makedirs(processed_dir, exist_ok=True)
    
    df = pd.read_csv(raw_path)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    
    y = df["Survived"].values
    X = df.drop(columns=["Survived"]).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Save as pickle files
    with open(f"{processed_dir}/train_features.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(f"{processed_dir}/train_targets.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(f"{processed_dir}/test_features.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(f"{processed_dir}/test_targets.pkl", "wb") as f:
        pickle.dump(y_test, f)
    
    print(f"Saved processed data to {processed_dir} as pickle files")

def titanic_dataset():
    import os
    # Get the project root (2 levels up from this file)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    processed_dir = os.path.join(project_root, "data", "processed")
    
    with open(os.path.join(processed_dir, "train_features.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(processed_dir, "train_targets.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(processed_dir, "test_features.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(processed_dir, "test_targets.pkl"), "rb") as f:
        y_test = pickle.load(f)
    
    return (
        torch.utils.data.TensorDataset(X_train, y_train),
        torch.utils.data.TensorDataset(X_test, y_test),
    )
class MyDataset(Dataset):
    """Minimal dataset placeholder to satisfy tests and simple use."""
    def __init__(self, root: str):
        self.root = root
        self.samples = []
        csv_path = os.path.join(root, "Titanic-Dataset.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                self.samples = df.to_dict("records")
            except Exception:
                self.samples = []
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    typer.run(preprocess_data)