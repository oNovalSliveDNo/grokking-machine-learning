# chapter03/utils/plotting.py


import numpy as np
import matplotlib.pyplot as plt
from models.polynomial_regression import predict_polynomial


def draw_line(slope, intercept, color='red', linewidth=0.7, start=0, end=8):
    """
    Рисует прямую линию по уравнению y = mx + b

    Parameters:
        slope (float): Угловой коэффициент (наклон) линии
        intercept (float): Свободный член (сдвиг по y)
        color (str): Цвет линии
        linewidth (float): Толщина линии
        start (float): Начало диапазона по оси x
        end (float): Конец диапазона по оси x
    """
    x = np.linspace(start, end, 1000)  # 1000 точек между start и end
    y = slope * x + intercept  # Вычисляем y по формуле
    plt.plot(x, y, linestyle='-', color=color, linewidth=linewidth)


def plot_points(features, labels):
    """
    Строит точки обучающей выборки (фичи и значения)

    Parameters:
        features (array-like): Массив значений x (например, количество комнат)
        labels (array-like): Массив значений y (например, цены)
    """
    plt.scatter(features, labels, color='blue')
    plt.xlabel('Количество комнат')
    plt.ylabel('Цена')
    plt.grid(True, linestyle='--', alpha=0.7)


def plot_errors(errors_list, error_name):
    """
    Строит график ошибок по эпохам обучения

    Parameters:
        errors_list (list): Список значений ошибки (например, RMSE на каждой эпохе)
        error_name (str): Название ошибки (для отображения в заголовке)
    """
    plt.scatter(range(len(errors_list)), errors_list, s=8)
    plt.title(f"{error_name.upper()} по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel(error_name.upper())
    plt.grid(True)
    plt.show()


def plot_model_poly(weights, degree, features, labels):
    """
    Строит график полиномиальной модели и точек выборки

    Parameters:
        weights (list or np.array): Коэффициенты полинома
        degree (int): Степень полинома
        features (array-like): Массив входных значений x
        labels (array-like): Массив фактических значений y
    """
    x_line = np.linspace(min(features), max(features), 100)  # Плавная линия по x
    y_line = predict_polynomial(weights, x_line, degree)  # Вычисляем y по полиному

    plt.plot(x_line, y_line, label="Модель", color="green")  # Линия модели
    plt.scatter(features, labels, label="Точки", color="blue")  # Исходные точки
    plt.title(f"Полиномиальная регрессия (degree={degree})")
    plt.legend()
    plt.grid(True)
    plt.show()
