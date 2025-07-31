# chapter03/utils/errors


import numpy as np


def mae(labels, predictions):
    """MAE — Средняя абсолютная ошибка.

    Шаги:
    1. Вычислить абсолютную разницу между фактическими и предсказанными значениями.
    2. Взять среднее значение этих разниц.
    """
    return np.mean(np.abs(labels - predictions))


def mse(labels, predictions):
    """MSE — Средняя квадратичная ошибка.

    Шаги:
    1. Найти разницу между фактическими и предсказанными значениями.
    2. Возвести эти разницы в квадрат.
    3. Взять среднее значение квадратов ошибок.
    """
    return np.mean((labels - predictions) ** 2)


def rmse(labels, predictions):
    """RMSE — Корень из средней квадратичной ошибки.

    Шаги:
    1. Найти MSE.
    2. Взять квадратный корень из этого значения.
    """
    return np.sqrt(np.mean((labels - predictions) ** 2))
