import numpy as np
import pandas as pd
from typing import Union,Dict, Any
from sklearn.model_selection import train_test_split

def calculate_label_proportions(labels: np.ndarray) -> Dict[Any, float]:
    """
    Calculate the proportions of each unique label in a NumPy array.

    Parameters:
        - labels (np.ndarray): NumPy array of target labels.

    Returns:
        - label_proportions (Dict[Any, float]): Dictionary with labels as keys and their proportions (rounded to 2 decimal places) as values.
    """
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    label_proportions = {label: round(count / total_samples, 2) for label, count in zip(unique_labels, label_counts)}
    
    return label_proportions


def sample_and_split(df: pd.DataFrame, target: str, n: int, random_seed: int) -> Union[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Randomly sample and split a DataFrame into training and testing sets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        target (str): The column name representing the target variable.
        n (int): The number of samples to be included in the random sample.
        random_seed (int): Seed for reproducibility in random sampling.

    Returns:
        X_train (pd.DataFrame): The feature data for training.
        X_test (pd.DataFrame): The feature data for testing.
        y_train (pd.Series): The target variable for training.
        y_test (pd.Series): The target variable for testing.
    """
    df_samp = df.sample(n=n, random_state=random_seed)
    X = df_samp.drop(target, axis=1)
    y = df_samp[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)

    return X_train, X_test, y_train, y_test





