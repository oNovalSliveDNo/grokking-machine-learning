# chapter05/utils/plotting.py


import matplotlib.pyplot as plt
import numpy as np


def plot_points(features, labels, x, y, ax=None, zorder=0):
    """
    Визуализирует точки данных на графике с разными маркерами для разных классов.

    Параметры:
        features (numpy.ndarray): Матрица признаков (не используется напрямую, но нужен для совместимости)
        labels (numpy.ndarray): Вектор меток классов (0 или 1)
        x (numpy.ndarray): Координаты точек по оси X
        y (numpy.ndarray): Координаты точек по оси Y
        ax (matplotlib.axes.Axes, optional): Объект осей для рисования. Если None, используется текущие оси.
        zorder (int, optional): Порядок отрисовки (чем больше, тем "выше" элемент). По умолчанию 0.
    """
    # Если оси не переданы, получаем текущие оси графика
    if ax is None:
        ax = plt.gca()

    # Рисуем точки для класса 0 (метка 0)
    # marker='s' - квадратные маркеры
    # s=300 - размер маркеров
    # c='blue' - синий цвет
    # label='Грустный (0)' - подпись для легенды
    ax.scatter(x[labels == 0], y[labels == 0],
               marker='s', s=300, c='blue', label='Грустный (0)', zorder=zorder)

    # Рисуем точки для класса 1 (метка 1)
    # marker='^' - треугольные маркеры
    # c='orange' - оранжевый цвет
    # label='Радостный (1)' - подпись для легенды
    ax.scatter(x[labels == 1], y[labels == 1],
               marker='^', s=300, c='orange', label='Радостный (1)', zorder=zorder)


def plot_learning_history(weights_history, bias_history, features, ax=None, zorder=0):
    """
    Визуализирует историю обучения, отображая все промежуточные разделяющие линии.

    Параметры:
        weights_history (list): Список векторов весов на каждом шаге обучения
        bias_history (list): Список значений смещения на каждом шаге обучения
        features (numpy.ndarray): Матрица признаков (используется для определения границ графика)
        ax (matplotlib.axes.Axes, optional): Объект осей для рисования. Если None, используется текущие оси.
        zorder (int, optional): Порядок отрисовки. По умолчанию 0.
    """
    # Если оси не переданы, получаем текущие оси графика
    if ax is None:
        ax = plt.gca()

    # Определяем границы по оси X для рисования линий
    x_min = min(features[:, 0]) - 0.5  # Левая граница с отступом 0.5
    x_max = max(features[:, 0]) + 0.5  # Правая граница с отступом 0.5

    # Рисуем все промежуточные классификаторы из истории обучения
    for i, (weights, bias) in enumerate(zip(weights_history, bias_history)):
        # Вычисляем координаты для разделяющей линии
        # Уравнение линии: w0*x + w1*y + bias = 0 => y = -(w0*x + bias)/w1
        x_values = np.array([x_min, x_max])  # Точки на краях графика
        y_values = -(weights[0] * x_values + bias) / weights[1]

        # Рисуем линию с полупрозрачным серым цветом и тонкой линией
        ax.plot(x_values, y_values,
                color='gray',  # Серый цвет
                alpha=0.1,  # Прозрачность 10%
                linewidth=1,  # Толщина линии 1
                zorder=zorder)  # Порядок отрисовки

    # Последнюю линию рисуем более заметной (если история не пустая)
    if len(weights_history) > 0:
        final_weights = weights_history[-1]  # Последние веса
        final_bias = bias_history[-1]  # Последнее смещение
        y_values = -(final_weights[0] * x_values + final_bias) / final_weights[1]
        ax.plot(x_values, y_values,
                color='gray',  # Серый цвет
                alpha=0.3,  # Прозрачность 30%
                linewidth=2,  # Толщина линии 2
                label='История обучения')  # Подпись для легенды


def plot_classifier(weights, bias, features, labels, x, y, weights_history=None, bias_history=None, ax=None):
    """
    Основная функция для визуализации классификатора и данных.

    Параметры:
        weights (numpy.ndarray): Вектор весов обученной модели
        bias (float): Смещение обученной модели
        features (numpy.ndarray): Матрица признаков
        labels (numpy.ndarray): Вектор меток классов
        x (numpy.ndarray): Координаты точек по оси X
        y (numpy.ndarray): Координаты точек по оси Y
        weights_history (list, optional): История весов в процессе обучения. По умолчанию None.
        bias_history (list, optional): История смещений в процессе обучения. По умолчанию None.
        ax (matplotlib.axes.Axes, optional): Объект осей для рисования. По умолчанию None.
    """
    # Если оси не переданы, получаем текущие оси графика
    if ax is None:
        ax = plt.gca()

    # 1. Сначала рисуем историю линий (если передана)
    if weights_history is not None and bias_history is not None:
        plot_learning_history(weights_history, bias_history, features, ax=ax, zorder=0)

    # 2. Затем рисуем точки данных (чтобы они были поверх линий истории)
    plot_points(features, labels, x, y, ax=ax, zorder=1)

    # 3. И только потом рисуем финальную линию классификатора (чтобы она была поверх всего)
    x_values = np.array([min(x) - 0.5, max(x) + 0.5])  # От края до края с отступами
    y_values = -(weights[0] * x_values + bias) / weights[1]  # Уравнение разделяющей линии

    # Рисуем финальную линию классификатора
    # 'k-' - черная сплошная линия
    # linewidth=4 - толщина линии 4
    # label='Финальный классификатор' - подпись для легенды
    # zorder=2 - рисуем поверх других элементов
    ax.plot(x_values, y_values, 'k-', linewidth=4, label='Финальный классификатор', zorder=2)

    # Настраиваем границы графика
    ax.set_xlim(min(x) - 0.5, max(x) + 0.5)  # Границы по X с отступами
    ax.set_ylim(min(y) - 0.5, max(y) + 0.5)  # Границы по Y с отступами


def plot_error_history(errors_list):
    """
    Визуализирует график изменения ошибки в процессе обучения.

    Параметры:
        errors_list (list): Список значений ошибки на каждом шаге обучения
    """
    # Рисуем график ошибки
    # 'b-' - синяя сплошная линия
    # linewidth=2 - толщина линии 2
    plt.plot(errors_list, 'b-', linewidth=2)
