# chapter09/utils/plotting.py


import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


def plot_points(dataset, x_col='x_0', y_col='x_1', target_col='y',
                class_names=('Class 0', 'Class 1'),
                colors=('blue', 'red'), markers=('s', '^'),
                title='Классификация точек', xlabel='x0', ylabel='x1',
                xlim=(0, 10), ylim=(0, 11), figsize=(12, 6), ax=None):
    """
    Визуализирует точки данных из DataFrame с разделением по классам.

    Параметры:
    - dataset: DataFrame с данными
    - x_col: название столбца с данными по оси X
    - y_col: название столбца с данными по оси Y
    - target_col: название столбца с метками классов
    - class_names: подписи классов для легенды
    - colors: цвета для каждого класса
    - markers: маркеры для каждого класса
    - title: заголовок графика
    - xlabel: подпись оси X
    - ylabel: подпись оси Y
    - xlim: пределы оси X
    - ylim: пределы оси Y
    - figsize: размер фигуры (используется только если ax=None)
    - ax: ось matplotlib для отрисовки (если None, создается новая фигура)
    """
    # Разделение данных по меткам
    classes = sorted(dataset[target_col].unique())
    class_data = [dataset[dataset[target_col] == cls] for cls in classes]

    # Создание графика, если не передана ось
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # Отображение точек для каждого класса
    for i, cls in enumerate(classes):
        ax.scatter(class_data[i][x_col], class_data[i][y_col],
                   marker=markers[i], s=100,
                   label=class_names[i],
                   color=colors[i],
                   edgecolor='black')

    # Настройка осей и заголовка
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend()

    # Если создавали новую фигуру, показываем ее
    if ax is None:
        plt.show()

    return ax


def plot_decision_boundary(model, features, labels, title='Decision Boundary',
                           figsize=(8, 6), ax=None, class_names=None,
                           colors=('blue', 'red'), markers=('s', '^')):
    """
    Визуализирует границу решения классификатора.

    Параметры:
    ----------
    model : обученная модель классификатора
    features : DataFrame или массив признаков (X)
    labels : Series или массив меток (y)
    title : str, заголовок графика
    figsize : tuple, размер фигуры (используется только если ax=None)
    ax : ось matplotlib для отрисовки (если None, создается новая фигура)
    class_names : список названий классов
    colors : tuple цветов для классов (должны соответствовать plot_points)
    markers : tuple маркеров для классов (должны соответствовать plot_points)
    """
    # Создание графика, если не передана ось
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    # Проверяем тип features (DataFrame или numpy array)
    if hasattr(features, 'iloc'):  # Если это DataFrame
        x0 = features.iloc[:, 0]
        x1 = features.iloc[:, 1]
    else:  # Если numpy array
        x0 = features[:, 0]
        x1 = features[:, 1]

    # Создаем график границы решения
    DecisionBoundaryDisplay.from_estimator(
        model,
        features,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
        cmap='coolwarm',
        ax=ax
    )

    # Получаем уникальные классы в правильном порядке (0, 1, 2...)
    unique_classes = np.unique(labels) if hasattr(labels, 'dtype') else sorted(labels.unique())

    for i, class_label in enumerate(unique_classes):
        mask = (labels == class_label)
        label = f'Class {class_label}' if class_names is None else class_names[i]
        ax.scatter(x0[mask], x1[mask],
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   label=label,
                   edgecolor='black',
                   s=100)

    ax.set_xlabel(features.columns[0] if hasattr(features, 'columns') else 'Feature 1')
    ax.set_ylabel(features.columns[1] if hasattr(features, 'columns') else 'Feature 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if ax is None:
        plt.show()

    return ax
