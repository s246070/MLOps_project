import pytest
import os
from mlops_project.data import preprocess_data

@pytest.fixture(scope="session")
def processed_data():
    preprocess_data()
    return True