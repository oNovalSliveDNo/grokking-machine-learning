# chapter05/utils/errors.py


import numpy as np


def score(weights, bias, features):
    """
    Вычисляет взвешенную сумму признаков (score) для классификации.

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Вектор признаков одного примера

    Возвращает:
        float: Взвешенная сумма признаков плюс смещение
    """
    # np.dot вычисляет скалярное произведение векторов weights и features
    # к результату добавляется смещение bias
    return np.dot(features, weights) + bias


def step(x):
    """
    Ступенчатая функция активации.

    Параметры:
        x (float): Входное значение

    Возвращает:
        int: 1 если x >= 0, иначе 0
    """
    # Если входное значение x больше или равно 0, возвращаем 1
    if x >= 0:
        return 1
    # Иначе возвращаем 0
    else:
        return 0


def prediction(weights, bias, features):
    """
    Выполняет предсказание класса с помощью перцептрона.

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Вектор признаков одного примера

    Возвращает:
        int: Предсказанный класс (0 или 1)
    """
    # Сначала вычисляем score (взвешенную сумму)
    # Затем применяем ступенчатую функцию для получения предсказания
    return step(score(weights, bias, features))


def error(weights, bias, features, label):
    """
    Вычисляет ошибку классификации для одного примера.

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Вектор признаков одного примера
        label (int): Истинная метка класса (0 или 1)

    Возвращает:
        float: Ошибка классификации для данного примера
    """
    # Получаем предсказание модели для текущих весов и признаков
    pred = prediction(weights, bias, features)

    # Если предсказание совпадает с истинной меткой, ошибка равна 0
    if pred == label:
        return 0

    # Если предсказание неверное, ошибка равна абсолютному значению score
    # Это показывает, насколько "уверенно" модель ошиблась
    else:
        return np.abs(score(weights, bias, features))


def mean_perceptron_error(weights, bias, features, labels):
    """
    Вычисляет среднюю ошибку перцептрона на всем наборе данных.

    Параметры:
        weights (numpy.ndarray): Вектор весов модели
        bias (float): Смещение (bias) модели
        features (numpy.ndarray): Матрица признаков (каждая строка - один пример)
        labels (numpy.ndarray): Вектор истинных меток классов

    Возвращает:
        float: Средняя ошибка перцептрона на всем наборе данных
    """
    # Инициализируем суммарную ошибку
    total_error = 0

    # Проходим по всем примерам в наборе данных
    for i in range(len(features)):
        # Для каждого примера вычисляем ошибку и добавляем к общей сумме
        total_error += error(weights, bias, features[i], labels[i])

    # Возвращаем среднюю ошибку (сумма ошибок / количество примеров)
    return total_error / len(features)
