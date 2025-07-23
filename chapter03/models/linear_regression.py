# chapter03/models/linear_regression


import random
import numpy as np
from utils.errors import mae, mse, rmse


def simple_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    """Простой метод корректировки весов и смещения."""
    small_random_1 = learning_rate
    small_random_2 = learning_rate

    predicted_price = base_price + price_per_room * num_rooms

    if price > predicted_price and num_rooms > 0:
        price_per_room += small_random_1
        base_price += small_random_2

    if price > predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2

    if price < predicted_price and num_rooms > 0:
        price_per_room -= small_random_1
        base_price -= small_random_2

    if price < predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2

    return price_per_room, base_price


def absolute_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    """Метод, минимизирующий абсолютную ошибку."""
    predicted_price = base_price + price_per_room * num_rooms

    if price > predicted_price:
        price_per_room += learning_rate * num_rooms
        base_price += learning_rate

    else:
        price_per_room -= learning_rate * num_rooms
        base_price -= learning_rate

    return price_per_room, base_price


def square_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    """Метод, минимизирующий среднеквадратичную ошибку."""
    predicted_price = base_price + price_per_room * num_rooms  # вычисляет прогноз
    error = price - predicted_price
    base_price += learning_rate * error  # перемещает прямую
    price_per_room += learning_rate * num_rooms * error  # вращает прямую
    return price_per_room, base_price


def linear_regression(
        features,  # Входной массив признаков (например, количество комнат)
        labels,  # Целевые значения (например, цены)
        learning_rate=0.01,  # Скорость обучения — как сильно изменяются веса на каждом шаге
        epochs=1000,  # Количество эпох (итераций обучения)
        trick='square',  # Метод обновления весов: 'simple', 'absolute', 'square'
        error='rmse',  # Метрика ошибки: 'mae', 'mse', 'rmse'
        mode='sgd',  # Режим обучения: 'sgd', 'mini', 'batch'
        batch_size=2  # Размер мини-пакета (актуально только для 'mini')
):
    # Случайная инициализация коэффициентов: наклон (price_per_room) и смещение (base_price)
    price_per_room = random.random()
    base_price = random.random()

    # Список для хранения значений ошибки на каждой итерации
    errors_list = []

    # Словарь доступных стратегий обновления весов
    tricks = {
        'simple': simple_trick,  # простой подход
        'absolute': absolute_trick,  # абсолютный подход (минимизация MAE)
        'square': square_trick  # квадратический подход (минимизация MSE (классическая градиентная регрессия))
    }

    # Словарь доступных метрик ошибок
    errors = {
        'mae': mae,  # средняя абсолютная ошибка
        'mse': mse,  # средняя квадратичная ошибка
        'rmse': rmse  # корень из средней квадратичной ошибки
    }

    # Проверка: допустима ли выбранный подход обновления весов
    if trick not in tricks:
        raise ValueError("Только 'simple', 'absolute' или 'square'")

    # Проверка: допустима ли выбранная метрика ошибки
    if error not in errors:
        raise ValueError("Ошибка должна быть: 'mae', 'mse', или 'rmse'")

    # Основной цикл обучения
    for epoch in range(epochs):
        # Предсказания по всей выборке (для отслеживания ошибки)
        predictions = price_per_room * features + base_price

        # Вычисление текущей ошибки модели и её сохранение
        errors_list.append(errors[error](labels, predictions))

        # === Разные режимы градиентного спуска ===
        if mode == 'sgd':
            # === Стохастический градиентный спуск: обновление по одной случайной точке ===
            i = random.randint(0, len(features) - 1)  # случайный индекс
            x_i, y_i = features[i], labels[i]  # выбор признака и цели
            price_per_room, base_price = tricks[trick](
                base_price, price_per_room, x_i, y_i, learning_rate
            )

        elif mode == 'batch':
            # === Пакетный градиентный спуск: обновление по всем точкам ===
            for x_i, y_i in zip(features, labels):
                predicted = price_per_room * x_i + base_price
                error_i = y_i - predicted  # ошибка предсказания
                # Градиентный шаг обновления весов:
                base_price += learning_rate * error_i
                price_per_room += learning_rate * x_i * error_i

        elif mode == 'mini':
            # === Мини-батч градиентный спуск: обновление по случайной подгруппе точек ===
            indices = np.random.choice(len(features), batch_size, replace=False)  # случайный батч
            for i in indices:
                x_i, y_i = features[i], labels[i]
                price_per_room, base_price = tricks[trick](
                    base_price, price_per_room, x_i, y_i, learning_rate
                )
        else:
            # Если режим указан некорректно — выбрасываем ошибку
            raise ValueError("mode должен быть 'sgd', 'batch' или 'mini'")

    # Возвращаем обученные параметры и список ошибок на всех итерациях
    return price_per_room, base_price, errors_list


def predict(price_per_room, base_price, rooms_count):
    return price_per_room * rooms_count + base_price
