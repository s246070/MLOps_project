"""
Example usage of ModelVisualizer for plotting training metrics and model analysis.

This demonstrates how to:
1. Fetch training history from W&B
2. Create various plots (training history, confusion matrix, ROC curve, feature importance)
3. Save plots locally and/or log them back to W&B
"""

import numpy as np
import wandb
from mlops_project.visualize import ModelVisualizer


def example_with_wandb_history():
    """Fetch real training history from W&B and plot it."""
    # Initialize W&B API
    api = wandb.Api()
    
    # Replace with your actual run path from W&B dashboard
    run_path = "s246070-danmarks-tekniske-universitet-dtu/titanic/eq60vev6"
    run = api.run(run_path)
    
    # Fetch history as pandas DataFrame
    history_df = run.history()
    
    # Convert to dict format expected by ModelVisualizer
    history = {
        "train_loss": history_df["train_loss"].tolist(),
        "val_accuracy": history_df["val_accuracy"].tolist(),
    }
    
    # Create visualizer and plot
    viz = ModelVisualizer(figsize=(12, 6))
    fig = viz.plot_training_history(
        history,
        metrics=["train_loss", "val_accuracy"],
        save_path="reports/figures/training_history.png",
        show=False  # Set to True to display interactively
    )
    
    # Optional: Log back to W&B
    wandb.init(project="titanic", job_type="visualization")
    wandb.log({"training_history": wandb.Image(fig)})
    wandb.finish()


def example_confusion_matrix():
    """Example confusion matrix plot."""
    # Example confusion matrix (replace with actual predictions)
    cm = np.array([
        [85, 12],   # True negatives, False positives
        [15, 67]    # False negatives, True positives
    ])
    
    viz = ModelVisualizer()
    fig = viz.plot_confusion_matrix(
        cm,
        class_names=["Did not survive", "Survived"],
        save_path="reports/figures/confusion_matrix.png",
        show=True
    )


def example_roc_curve():
    """Example ROC curve plot."""
    # Example ROC data (replace with actual values from sklearn.metrics.roc_curve)
    fpr = np.array([0.0, 0.05, 0.15, 0.25, 1.0])
    tpr = np.array([0.0, 0.70, 0.85, 0.92, 1.0])
    roc_auc = 0.87
    
    viz = ModelVisualizer()
    fig = viz.plot_roc_curve(
        fpr,
        tpr,
        roc_auc,
        save_path="reports/figures/roc_curve.png",
        show=True
    )


def example_feature_importance():
    """Example feature importance plot."""
    # Example feature importances (replace with actual model coefficients)
    feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    importances = np.array([0.15, 0.45, 0.12, 0.08, 0.05, 0.10, 0.05])
    
    viz = ModelVisualizer()
    fig = viz.plot_feature_importance(
        importances,
        feature_names,
        top_n=7,
        save_path="reports/figures/feature_importance.png",
        show=True
    )


def example_with_mock_data():
    """Quick example with mock training data."""
    # Mock training history
    epochs = 100
    history = {
        "loss": [0.7 - 0.005 * i + np.random.random() * 0.05 for i in range(epochs)],
        "val_loss": [0.75 - 0.004 * i + np.random.random() * 0.06 for i in range(epochs)],
    }
    
    viz = ModelVisualizer(figsize=(10, 5))
    fig = viz.plot_training_history(
        history,
        metrics=["loss", "val_loss"],
        save_path="reports/figures/mock_training.png",
        show=True
    )


if __name__ == "__main__":
    example_with_wandb_history()
    example_confusion_matrix()
    example_roc_curve()
    example_feature_importance()
    example_with_mock_data()
