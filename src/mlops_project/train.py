import torch
import wandb
import os
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from mlops_project.data import titanic_dataset
from mlops_project.model import LogisticRegressionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train Titanic logistic regression with Hydra config."""

    # Print config for visibility
    print(OmegaConf.to_yaml(cfg))

    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)

    # Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Load data
    full_train_set, _ = titanic_dataset()
    train_size = int(cfg.data.train_val_split * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(
        full_train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_set, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=256)

    # Initialize model
    input_dim = full_train_set.tensors[0].shape[1]
    model = LogisticRegressionModel(input_dim).to(DEVICE)

    # Loss
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    if cfg.hyperparameters.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.hyperparameters.lr,
            weight_decay=cfg.hyperparameters.weight_decay,
        )
    elif cfg.hyperparameters.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.hyperparameters.lr,
            weight_decay=cfg.hyperparameters.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.hyperparameters.optimizer}")

    # Training loop
    for epoch in range(cfg.hyperparameters.epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.float().to(DEVICE).squeeze()
            logits = model(x).squeeze()
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == y.bool()).sum().item()
            total += y.numel()

        train_loss /= len(train_loader)
        train_accuracy = correct / total if total > 0 else 0.0

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.float().to(DEVICE).squeeze()
                logits = model(x).squeeze()
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = torch.sigmoid(logits) > 0.5
                correct += (preds == y.bool()).sum().item()
                total += y.numel()

        val_loss /= len(val_loader)
        val_accuracy = correct / total if total > 0 else 0.0

        # ---- Log metrics ----
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # ---- Save model ----
    model_dir = Path(cfg.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Gem modellen med unik timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    local_path = f"models/modelweights_{timestamp}.pth"
    # Save state_dict with input_dim metadata for reconstruction in Cloud Functions
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
    }, local_path)

    # Upload til GCS med samme unikke navn
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("mlops-project-models")
    blob_name = f"modelweights_{timestamp}.pth"
    bucket.blob(blob_name).upload_from_filename(local_path)
    print(f"Upload complete: {blob_name}")

    run.finish()


if __name__ == "__main__":
    train()
