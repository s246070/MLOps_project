import torch
from mlops_project.model import LogisticRegressionModel

def test_one_training_step_runs():
    input_dim = 7
    model = LogisticRegressionModel(input_dim)

    x = torch.randn(8, input_dim)
    y = torch.randint(0, 2, (8, 1)).float()  # binær target

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # forward pass and loss computation
    logits = model(x)
    loss = criterion(logits, y)
    # backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # check that loss is finite
    assert torch.isfinite(loss).item() is True