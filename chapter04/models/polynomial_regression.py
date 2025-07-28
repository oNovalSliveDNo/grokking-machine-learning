# chapter04/models/polynomial_regression.py


import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def train_polynomial_regression(features, labels, degree):
    """
    Обучает модель полиномиальной регрессии заданной степени.

    Параметры:
    features (array-like): Входные признаки (независимые переменные)
    labels (array-like): Целевые значения (зависимая переменная)
    degree (int): Степень полинома для преобразования признаков

    Возвращает:
    tuple: (Обученная модель, Полиномиальный преобразователь)
    """
    # Создаем преобразователь полиномиальных признаков
    poly = PolynomialFeatures(degree)

    # Преобразуем исходные признаки в полиномиальные
    # Например, для degree=2: [x] -> [1, x, x^2]
    features_poly = poly.fit_transform(features)

    # Создаем и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(features_poly, labels)

    # Возвращаем обученную модель и преобразователь (чтобы использовать его на новых данных)
    return model, poly


def evaluate_model(model, features_poly, labels):
    """
    Вычисляет метрики качества для обученной модели.

    Параметры:
    model: Обученная модель регрессии
    features_poly (array-like): Полиномиальные признаки
    labels (array-like): Истинные целевые значения

    Возвращает:
    tuple: (MAE, MSE, RMSE) - метрики ошибок
    """
    # Получаем предсказания модели
    predictions = model.predict(features_poly)

    # Вычисляем метрики:
    # MAE (Mean Absolute Error) - средняя абсолютная ошибка
    mae = mean_absolute_error(labels, predictions)

    # MSE (Mean Squared Error) - средняя квадратичная ошибка
    mse = mean_squared_error(labels, predictions)

    # RMSE (Root Mean Squared Error) - корень из средней квадратичной ошибки
    rmse = np.sqrt(mse)

    return mae, mse, rmse


def train_polynomial_regression_regularized(features, labels, degree, penalty='l2', alpha=1.0):
    """
    Обучает полиномиальную регрессию с регуляризацией (L1 или L2).

    Параметры:
    features (array-like): Входные признаки
    labels (array-like): Целевые значения
    degree (int): Степень полинома
    penalty (str): Тип регуляризации ('l1' для Lasso, 'l2' для Ridge)
    alpha (float): Сила регуляризации (больше значение → сильнее регуляризация)

    Возвращает:
    tuple: (Обученная модель, Полиномиальный преобразователь)

    Выбрасывает:
    ValueError: Если указан недопустимый тип регуляризации
    """
    # Создаем преобразователь полиномиальных признаков
    poly = PolynomialFeatures(degree)

    # Преобразуем исходные признаки в полиномиальные
    features_poly = poly.fit_transform(features)

    # Выбираем тип модели в зависимости от типа регуляризации
    if penalty == 'l1':
        # Lasso регрессия (L1 регуляризация) - создает разреженные модели
        # Увеличиваем max_iter для гарантии сходимости
        model = Lasso(alpha=alpha, max_iter=10000)
    elif penalty == 'l2':
        # Ridge регрессия (L2 регуляризация) - уменьшает веса коэффициентов
        model = Ridge(alpha=alpha)
    else:
        # Обработка недопустимых значений
        raise ValueError("penalty должен быть 'l1' или 'l2'")

    # Обучаем модель на полиномиальных признаках
    model.fit(features_poly, labels)

    return model, poly
