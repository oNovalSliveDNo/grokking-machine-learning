# chapter06/utils/errors.py

import numpy as np


def sigmoid(x):
    """
    Вычисляет значение сигмоидной функции.

    Параметры:
        x (float): Входное значение

    Возвращает:
        float: Значение сигмоидной функции в точке x
    """
    # Численно стабильная реализация сигмоиды:
    # Если x >= 0, используем стандартную формулу 1 / (1 + e^(-x))
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        # Если x < 0, используем эквивалентную форму для избежания переполнения
        return np.exp(x) / (1 + np.exp(x))


def score(weights, bias, features):
    """
    Вычисляет взвешенную сумму признаков (линейную комбинацию).

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Вектор признаков одного примера

    Возвращает:
        float: Взвешенная сумма признаков плюс смещение
    """
    # Скалярное произведение признаков и весов + смещение
    return np.dot(weights, features) + bias


def prediction(weights, bias, features):
    """
    Выполняет предсказание вероятности положительного класса.

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Вектор признаков одного примера

    Возвращает:
        float: Предсказанная вероятность принадлежности к классу 1
    """
    # Вычисляем score и передаём в сигмоиду для получения вероятности
    return sigmoid(score(weights, bias, features))


def log_loss(weights, bias, features, label):
    """
    Вычисляет логарифмическую потерю (log loss) для одного примера.

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Вектор признаков одного примера
        label (int): Истинная метка класса (0 или 1)

    Возвращает:
        float: Логарифмическая потеря для данного примера
    """
    # Вычисляем предсказанную вероятность
    pred = prediction(weights, bias, features)

    # Применяем формулу логарифмической потери:
    # -y * log(p) - (1 - y) * log(1 - p)
    return - label * np.log(pred) - (1 - label) * np.log(1 - pred)


def total_log_loss(weights, bias, features, labels):
    """
    Вычисляет общую логарифмическую потерю по всему набору данных.

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Матрица признаков (каждая строка — один пример)
        labels (numpy.ndarray): Вектор истинных меток классов

    Возвращает:
        float: Суммарная логарифмическая потеря на всем наборе данных
    """
    # Инициализируем суммарную ошибку
    total_error = 0

    # Проходим по всем примерам и накапливаем потери
    for i in range(len(features)):
        total_error += log_loss(weights, bias, features[i], labels[i])

    # Возвращаем суммарную (не усреднённую) логарифмическую потерю
    return total_error
