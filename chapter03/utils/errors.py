import numpy as np


def rmse(labels, predictions):
    """Корень из средней квадратичной ошибки."""
    differences = np.subtract(labels, predictions)
    return np.sqrt(np.mean(differences ** 2))
