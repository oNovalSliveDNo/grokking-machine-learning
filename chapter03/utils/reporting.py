# chapter03/utils/reporting.py


import numpy as np
from models.polynomial_regression import predict_polynomial


def format_equation(slope, intercept, precision=2):
    """
    Форматирует уравнение прямой вида y = mx + b.

    Parameters:
        slope (float): Наклон прямой.
        intercept (float): Смещение (свободный член).
        precision (int): Количество знаков после запятой.

    Returns:
        str: Строковое представление уравнения.
    """
    m = round(slope, precision)  # Округляем наклон до нужной точности
    b = round(intercept, precision)  # Округляем смещение до нужной точности
    sign = "+" if b >= 0 else "-"  # Выбираем знак между m*x и b
    # Возвращаем строку вида: y = 2.0 * x + 5.0
    return f"y = {m} * x {sign} {abs(b)}"


def print_prediction(slope, intercept, x_value):
    """
    Формирует строку с предсказанием для линейной модели.

    Parameters:
        slope (float): Наклон прямой.
        intercept (float): Свободный член.
        x_value (float or int): Значение x, для которого делается прогноз.

    Returns:
        str: Строка с результатом предсказания.
    """
    y = slope * x_value + intercept  # Вычисляем предсказанное значение
    return f"Для {x_value} комнат → Предсказанная цена: {y:.2f}"


def format_polynomial_equation(weights, precision=2):
    """
    Формирует уравнение полинома по вектору весов.

    Parameters:
        weights (np.ndarray): Коэффициенты полинома (от x^0 до x^n).
        precision (int): Количество знаков после запятой.

    Returns:
        str: Форматированное уравнение вида y = w0 + w1*x^1 + ...
    """
    # Формируем отдельные элементы уравнения для каждого члена полинома
    terms = [
        f"({round(w, precision)} * x^{i})" if i > 0 else f"{round(w, precision)}"
        for i, w in enumerate(weights)
    ]
    # Объединяем все члены в одну строку через " + "
    return "y = " + " + ".join(terms)


def print_prediction_poly(weights, degree, x_value, precision=2):
    """
    Делает предсказание и форматирует его для полиномиальной регрессии.

    Parameters:
        weights (np.ndarray): Коэффициенты полинома.
        degree (int): Степень полинома.
        x_value (float or int): Значение x, для которого делается прогноз.
        precision (int): Кол-во знаков после запятой для результата.

    Returns:
        str: Строка с результатом предсказания.
    """
    # Вызываем функцию предсказания, подаём x как массив из одного элемента
    y_pred = predict_polynomial(weights, np.array([x_value]), degree)[0]
    # Возвращаем красиво отформатированный результат
    return f"Для x = {x_value} → Предсказанное значение: {y_pred:.{precision}f}"
