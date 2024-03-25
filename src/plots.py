import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import math
from typing import Union, List


def plot_residuals(y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray]) -> None:
    """
    Plot a residuals scatter plot.

    Parameters:
        y_true : Actual values.
        y_pred : Predicted values.
    """

    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.title('Residuals Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()


def norm_confusion_plot(y_true: List[int], y_pred: List[int], labels: List[int] = None) -> None:
    """
    Plot a normalized confusion matrix.

    Parameters:
        y_true : Actual values.
        y_pred : Predicted values.
        labels : Target feature values.
    """
    mx = confusion_matrix(y_true, y_pred, labels=labels)
    cmn = mx.astype('float') / mx.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cmn, cmap='Blues', interpolation='nearest')

    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{cmn[i, j]:.2f}', ha='center', va='center', color='black')

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.show()

def boxplots(df: pd.DataFrame, x_column: str, y_columns: List[str]):
    """
    Plot a subplot of boxplots.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_column (str): The column name to be used for the x-axis.
        y_columns (List[str]): A list of column names to be used for the y-axis.
    """
    fig, axes = plt.subplots(len(y_columns), 1, figsize=(6, 12))

    for i, y_column in enumerate(y_columns):
        sns.boxplot(ax=axes[i], data=df, x=x_column, y=y_column)
        axes[i].set_title(f'Boxplot for {y_column}')

    plt.tight_layout()
    plt.show()
