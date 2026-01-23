import pytest
import torch

from mlops_project.model import LogisticRegressionModel


def test_model_initialization():
    input_dim = 7
    model = LogisticRegressionModel(input_dim)
    assert isinstance(model, LogisticRegressionModel)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_forward_output_shape(batch_size):
    # Check that forward pass produces correct output shape for different batch sizes
    input_dim = 7
    model = LogisticRegressionModel(input_dim)

    x = torch.randn(batch_size, input_dim)
    y = model(x)

    assert y.shape == (batch_size, 1)


def test_forward_with_wrong_input_dim_raises():
    # check that incorrect input dimension raises an error
    input_dim = 7
    model = LogisticRegressionModel(input_dim)

    x_wrong = torch.randn(4, input_dim + 1)

    with pytest.raises(RuntimeError):
        model(x_wrong)
