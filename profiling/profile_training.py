
import cProfile
import pstats
from pathlib import Path
import sys
from mlops_project.data_pickle import titanic_dataset
from mlops_project.model import LogisticRegressionModel
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def train_model():
    "Simple training function to profile"
    device = torch.device("cpu")
    
    # Load data
    full_train_set, _ = titanic_dataset()
    train_size = int(0.8 * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    
    # Model
    input_dim = full_train_set.tensors[0].shape[1]
    model = LogisticRegressionModel(input_dim).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    model.train()
    for epoch in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.float().to(device).squeeze()
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("profiling/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    train_model()
    profiler.disable()
    
    # Save stats for snakeviz
    stats_file = output_dir / "profile_stats.prof"
    profiler.dump_stats(str(stats_file))
    
    # Also print text summary
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    print("\n" + "="*80)
    print("✓ Profiling complete!")
    print("="*80)
    print("\nTo view interactive visualizatio run:")
    print(f"  snakeviz {stats_file}")
    print("\nThis will open in your browser charts")