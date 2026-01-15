import torch
from mlops_project.model import LogisticRegressionModel

def test_one_training_step_runs():
    input_dim = 7
    model = LogisticRegressionModel(input_dim)

    x = torch.randn(8, input_dim)
    y = torch.randint(0, 2, (8, 1)).float()  # binær target

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item() is True