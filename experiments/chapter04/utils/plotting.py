# chapter04/utils/plotting.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_polynomial_model_with_test(train_features, train_labels,
                                    test_features, test_labels,
                                    model, poly, degree, ax=None):
    """
    Визуализирует полиномиальную модель регрессии вместе с обучающими и тестовыми данными.

    Параметры:
    train_features (array-like): Признаки обучающей выборки
    train_labels (array-like): Целевые значения обучающей выборки
    test_features (array-like): Признаки тестовой выборки
    test_labels (array-like): Целевые значения тестовой выборки
    model: Обученная модель регрессии (должна поддерживать метод predict())
    poly: Полиномиальный преобразователь (должен поддерживать метод transform())
    degree (int): Степень полинома
    ax (matplotlib.axes.Axes, optional): Ось для отрисовки. Если не задана, создается новая фигура.
    """
    # Создаем диапазон значений x для предсказания
    # Берем минимальное и максимальное значение из обеих выборок
    x_min = min(train_features.min(), test_features.min())
    x_max = max(train_features.max(), test_features.max())

    # Генерируем 200 равномерно распределенных точек в этом диапазоне
    x_range = np.linspace(x_min, x_max, 200).reshape(-1, 1)

    # Преобразуем точки в полиномиальные признаки
    x_range_poly = poly.transform(x_range)

    # Получаем предсказания модели для всего диапазона
    y_range_pred = model.predict(x_range_poly)

    # Создаем новую фигуру, если ось не передана
    if ax is None:
        fig, ax = plt.subplots()

    # Отрисовываем кривую полиномиальной регрессии
    ax.plot(x_range, y_range_pred, color='red', label=f"Полином степени {degree}")

    # Отображаем обучающие точки синим цветом
    ax.scatter(train_features, train_labels, color='blue', label="Train")

    # Отображаем тестовые точки зеленым цветом с крестиками
    ax.scatter(test_features, test_labels, color='green', marker='x', s=80, label="Test")

    # Устанавливаем границы осей
    ax.set_xlim([-8, 8])
    ax.set_ylim([-5, 8])

    # Подписываем оси
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Добавляем заголовок с указанием степени полинома
    ax.set_title(f"Полином. регрессия (степень {degree})")

    # Включаем сетку
    ax.grid(True)

    # Добавляем легенду
    ax.legend()


def plot_mae_vs_degree_with_table(degrees, train_maes, test_maes):
    """
    Визуализирует зависимость MAE от степени полинома с таблицей значений.

    Параметры:
    degrees (list): Список степеней полиномов
    train_maes (list): Значения MAE на обучающей выборке для каждой степени
    test_maes (list): Значения MAE на тестовой выборке для каждой степени
    """
    # Создаем DataFrame для хранения результатов
    results_df = pd.DataFrame({
        "Степень полинома": degrees,
        "Train MAE": train_maes,
        "Test MAE": test_maes
    })

    # Создаем фигуру с двумя областями: для графика и таблицы
    # Соотношение ширины 2:1 (график в два раза шире таблицы)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    # График MAE в первой области (ax1)
    # Обучающая ошибка - синяя линия с кружками
    ax1.plot(degrees, train_maes, marker='o', label='Train MAE')

    # Тестовая ошибка - оранжевая линия с квадратами
    ax1.plot(degrees, test_maes, marker='s', label='Test MAE')

    # Подписываем оси
    ax1.set_xlabel('Степень полинома')
    ax1.set_ylabel('MAE')

    # Добавляем заголовок
    ax1.set_title('MAE vs. степень полинома')

    # Устанавливаем метки на оси X по всем степеням
    ax1.set_xticks(degrees)

    # Добавляем легенду
    ax1.legend()

    # Включаем сетку
    ax1.grid(True)

    # Таблица со значениями во второй области (ax2)
    # Отключаем оси для области с таблицей
    ax2.axis('off')

    # Создаем таблицу с округленными значениями
    table = ax2.table(
        cellText=np.round(results_df.values, 2),  # Округляем значения до 2 знаков
        colLabels=results_df.columns,  # Заголовки столбцов
        loc='center',  # Размещение по центру
        cellLoc='center'  # Выравнивание текста в ячейках
    )

    # Настраиваем размер шрифта
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Масштабируем таблицу для лучшего отображения
    table.scale(1.2, 1.5)

    # Убираем лишние пробелы вокруг графиков
    plt.tight_layout()

    # Отображаем фигуру
    plt.show()
