# import torch
# import wandb
# #from dotenv import load_dotenv
# import os
# import hydra
# from omegaconf import DictConfig, OmegaConf
# from pathlib import Path
# from torch.utils.data import DataLoader, random_split

# from mlops_project.data_pickle import titanic_dataset
# from mlops_project.model import LogisticRegressionModel

# #load_dotenv()
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @hydra.main(version_base=None, config_path="../../configs", config_name="config")
# def train(cfg: DictConfig) -> None:
#     """Train model with Hydra configuration."""
    
#     #print config for visibiliy 
#     print(OmegaConf.to_yaml(cfg))
    
#     #Seed for reproducability
#     torch.manual_seed(cfg.seed)
    
#     #Initialize wandb
#     wandb.login(key=os.getenv("WANDB_API_KEY"))
#     run = wandb.init(
#         project=cfg.wandb.project,
#         entity=cfg.wandb.entity,
#         config=OmegaConf.to_container(cfg, resolve=True),
#     )
    
#     # Load data
#     full_train_set, test_set = titanic_dataset()
#     train_size = int(cfg.data.train_val_split * len(full_train_set))
#     val_size = len(full_train_set) - train_size
#     train_set, val_set = random_split(
#         full_train_set, 
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(cfg.seed)
#     )
    
#     train_loader = DataLoader(
#         train_set, 
#         batch_size=cfg.hyperparameters.batch_size, 
#         shuffle=True
#     )
#     val_loader = DataLoader(val_set, batch_size=256)
    
#     # Initialize model
#     input_dim = full_train_set.tensors[0].shape[1]
#     model = LogisticRegressionModel(input_dim).to(DEVICE)
    
#     # Loss and optimizer
#     criterion = torch.nn.BCEWithLogitsLoss()
    
#     if cfg.hyperparameters.optimizer == "adam":
#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=cfg.hyperparameters.lr,
#             weight_decay=cfg.hyperparameters.weight_decay,
#         )
#     elif cfg.hyperparameters.optimizer == "sgd":
#         optimizer = torch.optim.SGD(
#             model.parameters(),
#             lr=cfg.hyperparameters.lr,
#             weight_decay=cfg.hyperparameters.weight_decay,
#         )
#     else:
#         raise ValueError(f"Unsupported optimizer: {cfg.hyperparameters.optimizer}")
    
#     # Training loop
#     for epoch in range(cfg.hyperparameters.epochs):
#         # Training
#         model.train()
#         epoch_loss = 0.0
#         for x, y in train_loader:
#             x, y = x.to(DEVICE), y.float().to(DEVICE).squeeze()
#             logits = model(x).squeeze()
#             loss = criterion(logits, y)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
        
#         epoch_loss /= len(train_loader)
        
#         # Validation
#         model.eval()
#         correct, total = 0, 0
#         val_loss = 0.0
#         with torch.no_grad():
#             for x, y in val_loader:
#                 x, y = x.to(DEVICE), y.float().to(DEVICE).squeeze()
#                 logits = model(x).squeeze()
#                 loss = criterion(logits, y)
#                 val_loss += loss.item()
                
#                 preds = torch.sigmoid(logits) > 0.5
#                 correct += (preds == y.bool()).sum().item()
#                 total += y.numel()
        
#         val_loss /= len(val_loader)
#         val_accuracy = correct / total if total > 0 else 0.0
        
#         # Log metrics
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": epoch_loss,
#             "val_loss": val_loss,
#             "val_accuracy": val_accuracy
#         })
        
#         # Print progress every 50 epochs
#         if (epoch + 1) % 50 == 0:
#             print(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs} - "
#                   f"Train Loss: {epoch_loss:.4f}, "
#                   f"Val Loss: {val_loss:.4f}, "
#                   f"Val Acc: {val_accuracy:.4f}")
    
#     # Save model
#     model_dir = Path(cfg.paths.model_dir)
#     model_dir.mkdir(parents=True, exist_ok=True)
    
#     # FJERNET FOR AT TESTE DRIFT DETECTION
#     #model_path = model_dir / "modelweights.pth"
#     #torch.save(model.state_dict(), model_path)
#     #print(f"Model saved to {model_path}")
    
#     # Save as pickle for cloud compatibility
#     import pickle
#     pkl_path = model_dir / "modelweights.pkl"
#     with open(pkl_path, "wb") as f:
#         pickle.dump(model.state_dict(), f)
#     print(f"Model saved to {pkl_path}")
    
#     run.finish()


# if __name__ == "__main__":
#     train()





# import torch
# import wandb
# from dotenv import load_dotenv
# import os
# from torch.utils.data import DataLoader, random_split
# from mlops_project.data import titanic_dataset
# from mlops_project.model import LogisticRegressionModel

# load_dotenv()

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def train(
#     lr: float = 0.01,
#     epochs: int = 500,
#     batch_size: int = 32,
#     optimizer_name: str = "adam",
#     weight_decay: float = 0.0,
# ):
#     wandb.login(key=os.getenv("_WANDB_KEY"))
#     run = wandb.init(
#         project="titanic", 
#         config={
#             "lr": lr,
#             "epochs": epochs,
#             "batch_size": batch_size,
#             "optimizer": optimizer_name,
#             "weight_decay": weight_decay,
#             },
#         )
#     config = wandb.config

#     full_train_set, _ = titanic_dataset()

#     train_size = int(0.8 * len(full_train_set))
#     val_size = len(full_train_set) - train_size
#     train_set, val_set = random_split(full_train_set, [train_size, val_size])

#     train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=256)

#     input_dim = full_train_set.tensors[0].shape[1]
#     model = LogisticRegressionModel(input_dim).to(DEVICE)

#     criterion = torch.nn.BCEWithLogitsLoss()

#     if config.optimizer == "adam":
#         optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=config.lr,
#         weight_decay=config.weight_decay,
#         )
#     elif config.optimizer == "sgd":
#         optimizer = torch.optim.SGD( 
#         model.parameters(),
#         lr=config.lr,
#         weight_decay=config.weight_decay,
#         )
#     else:
#         raise ValueError(f"Unsupported optimizer: {config.optimizer}")

#     for epoch in range(config.epochs):
#         # Training
#         model.train()
#         correct, total = 0, 0
#         epoch_loss = 0.0

#         for x, y in train_loader:
#             x, y = x.to(DEVICE), y.float().to(DEVICE).squeeze()
#             logits = model(x).squeeze()
#             loss = criterion(logits, y)
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
            
#             # Beregn korrekt klassificerede eksempler
#             preds = (torch.sigmoid(logits) > 0.5)
#             correct += (preds == y.bool()).sum().item()
#             total += y.numel()
            
#         epoch_loss /= len(train_loader)
#         train_accuracy = correct / total if total > 0 else 0.0
        
#         # Validation
#         model.eval()
#         correct, total = 0, 0
#         val_loss = 0.0

#         with torch.no_grad():
#             for x, y in val_loader:
#                 x = x.to(DEVICE)
#                 y = y.float().to(DEVICE).squeeze()
#                 logits = model(x).squeeze()
#                 loss = criterion(logits, y)
#                 val_loss += loss.item()
                
#                 preds = torch.sigmoid(logits) > 0.5
#                 correct += (preds == y.bool()).sum().item()
#                 total += y.numel()
                
#         val_accuracy = correct / total if total > 0 else 0.0
#         val_loss /= len(val_loader)
        
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": epoch_loss,
#             "train_accuracy": train_accuracy,
#             "val_loss": val_loss,
#             "val_accuracy": val_accuracy
#         })

#     run.finish()
#     os.makedirs("models", exist_ok=True)

#     # Gem modellen med unik timestamp
#     import datetime
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     local_path = f"models/modelweights_{timestamp}.pth"
#     torch.save(model.state_dict(), local_path)

#     # Upload til GCS med samme unikke navn
#     from google.cloud import storage
#     client = storage.Client()
#     bucket = client.bucket("mlops-project-models")
#     blob_name = f"modelweights_{timestamp}.pth"
#     bucket.blob(blob_name).upload_from_filename(local_path)
#     print(f"Upload complete: {blob_name}")

# if __name__ == "__main__":
#     train()


# # Cloud build trigger test #2
