import seaborn as sns
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class ModelVisualizer:
    """Visualization utilities for ML models."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize
        sns.set_style("whitegrid")
    
    def plot_training_history(
        self,
        history: dict,
        metrics: list = ["loss", "val_loss"],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """Plot training and validation metrics over epochs."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for metric in metrics:
            if metric in history:
                ax.plot(history[metric], label=metric, linewidth=2)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.set_title("Training History")
        ax.legend()
        ax.grid(True)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: Optional[list] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """Plot confusion matrix as heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names if class_names else 'auto',
                    yticklabels=class_names if class_names else 'auto')
        
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title("Confusion Matrix")
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    def plot_roc_curve(
        self,
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    def plot_feature_importance(
        self,
        importances: np.ndarray,
        feature_names: list,
        top_n: int = 10,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Figure:
        """Plot feature importance."""
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.barh(range(len(indices)), importances[indices])
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig