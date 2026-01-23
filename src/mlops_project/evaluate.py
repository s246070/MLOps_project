# titanic/evaluate.py
import torch
import typer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from mlops_project.data import titanic_dataset
from mlops_project.model import LogisticRegressionModel

# use gpu if possible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_NAMES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]

def evaluate(model_path: str = "models/modelweights.pth"):
    run = wandb.init(
        project="titanic",
        job_type="evaluation",
    )

    # Load data
    train_set, _ = titanic_dataset()
    input_dim = train_set.tensors[0].shape[1]

    # Load model
    model = LogisticRegressionModel(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Extract coefficients
    weights = model.linear.weight.detach().cpu().numpy().flatten()
    bias = model.linear.bias.detach().cpu().numpy().item()

    df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "weight": weights,
        "odds_ratio": np.exp(weights),
    })

    wandb.log({
        "coefficients": wandb.Table(dataframe=df)
    })

    # Plot coefficients
    plt.figure(figsize=(8, 4))
    plt.barh(df["feature"], df["weight"])
    plt.axvline(0, color="black", linestyle="--")
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Weight")
    plt.tight_layout()

    wandb.log({
        "coefficients_plot": wandb.Image(plt)
    })

    plt.close()

    # Log bias separately
    wandb.log({"bias": bias})

    print("\nModel coefficients:")
    print(df)

    run.finish()

if __name__ == "__main__":
    typer.run(evaluate)
