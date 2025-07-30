# chapter03/models/polynomial_regression.py


import random
import numpy as np
from utils.errors import mae, mse, rmse


def square_trick_poly(weights, x, y, learning_rate):
    """
    Выполняет один шаг градиентного спуска для полиномиальной регрессии (с MSE).

    Parameters:
        weights (np.ndarray): Вектор весов модели.
        x (np.ndarray): Полиномиальные признаки одной обучающей точки.
        y (float): Истинное значение для этой точки.
        learning_rate (float): Скорость обучения.

    Returns:
        np.ndarray: Обновлённые веса модели.
    """
    predicted = np.dot(weights, x)  # Предсказание модели
    error = y - predicted  # Ошибка (истинное - предсказанное)
    gradients = learning_rate * error * x  # Градиент для всех весов
    return weights + gradients  # Обновлённые веса


def expand_polynomial_features(features, degree):
    """
    Преобразует входные признаки в полиномиальные до заданной степени.

    Parameters:
        features (np.ndarray): Одномерный массив входных признаков (x).
        degree (int): Степень полинома (например, 2 создаёт x^0, x^1, x^2).

    Returns:
        np.ndarray: Массив формы (n_samples, degree + 1) с полиномиальными признаками.
    """
    return np.array([x ** np.arange(degree + 1) for x in features])


def polynomial_regression(
        features,  # Входные признаки (например, количество комнат)
        labels,  # Целевые значения (например, цены)
        degree=2,  # Степень полинома
        learning_rate=0.01,  # Скорость обучения
        epochs=1000,  # Количество итераций обучения
        error='rmse',  # Выбранная метрика ошибки: 'mae', 'mse', 'rmse'
        mode='sgd',  # Режим обучения: 'sgd', 'mini', 'batch'
        batch_size=2  # Размер мини-батча (для режима 'mini')
):
    """
    Обучает модель полиномиальной регрессии с использованием градиентного спуска.

    Returns:
        tuple:
            - weights (np.ndarray): Обученные коэффициенты полинома.
            - errors_list (list): История ошибок на каждой эпохе.
    """
    # Шаг 1: расширение признаков до полиномиальных
    X_poly = expand_polynomial_features(features, degree)

    # Шаг 2: инициализация весов случайными значениями
    weights = np.random.rand(degree + 1)

    # Шаг 3: определение метрики ошибки
    errors = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

    if error not in errors:
        raise ValueError("Ошибка должна быть: 'mae', 'mse', или 'rmse'")

    # Список для отслеживания ошибки на каждой итерации
    errors_list = []

    # Шаг 4: цикл обучения
    for epoch in range(epochs):
        # Текущее предсказание модели по всей обучающей выборке
        predictions = np.dot(X_poly, weights)

        # Вычисляем и сохраняем ошибку по текущим весам
        errors_list.append(errors[error](labels, predictions))

        # === Выбор режима обучения ===
        if mode == 'sgd':
            # Стохастический градиентный спуск: обновляем по одной случайной точке
            i = random.randint(0, len(features) - 1)
            x_i = X_poly[i]
            y_i = labels[i]
            weights = square_trick_poly(weights, x_i, y_i, learning_rate)

        elif mode == 'mini':
            # Мини-батч: выбираем случайную подгруппу точек и обновляем веса по ним
            indices = np.random.choice(len(features), batch_size, replace=False)
            for i in indices:
                x_i = X_poly[i]
                y_i = labels[i]
                weights = square_trick_poly(weights, x_i, y_i, learning_rate)

        elif mode == 'batch':
            # Пакетный градиентный спуск: проходим по всем точкам
            for x_i, y_i in zip(X_poly, labels):
                weights = square_trick_poly(weights, x_i, y_i, learning_rate)

        else:
            # Некорректный режим обучения
            raise ValueError("mode должен быть 'sgd', 'batch' или 'mini'")

    # Возвращаем обученные веса и историю ошибок
    return weights, errors_list


def predict_polynomial(weights, features, degree):
    """
    Делает предсказание с помощью обученной полиномиальной модели.

    Parameters:
        weights (np.ndarray): Веса обученной модели.
        features (np.ndarray): Входные признаки (x).
        degree (int): Степень полинома, использованная при обучении.

    Returns:
        np.ndarray: Предсказанные значения.
    """
    X_poly = expand_polynomial_features(features, degree)  # Расширяем признаки
    return np.dot(X_poly, weights)  # Возвращаем y = Xw
