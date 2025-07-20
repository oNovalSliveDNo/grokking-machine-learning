import random


def simple_trick(base_price, price_per_room, num_rooms, price):
    """Простой метод корректировки весов и смещения."""
    small_random_1 = random.random() * 0.1
    small_random_2 = random.random() * 0.1

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


def linear_regression(features, labels, learning_rate=0.01, epochs=1000, trick='square'):
    # генерируем случайные значения для наклона и y-пересечения
    price_per_room = random.random()
    base_price = random.random()

    # повторяем шаг обновления много раз
    for epoch in range(epochs):
        i = random.randint(0, len(features) - 1)  # выбираем случайную точку в наборе данных

        num_rooms = features[i]
        price = labels[i]

        if trick == 'simple':
            # применяем простой подход
            price_per_room, base_price = simple_trick(base_price,
                                                      price_per_room,
                                                      num_rooms,
                                                      price,
                                                      learning_rate=learning_rate)
        elif trick == 'absolute':
            # применяем абсолютный подход
            price_per_room, base_price = absolute_trick(base_price,
                                                        price_per_room,
                                                        num_rooms,
                                                        price,
                                                        learning_rate=learning_rate)
        elif trick == 'square':
            # применяем квадратический подход
            price_per_room, base_price = square_trick(base_price,
                                                      price_per_room,
                                                      num_rooms,
                                                      price,
                                                      learning_rate=learning_rate)
    return price_per_room, base_price


def predict(price_per_room, base_price, rooms_count):
    return price_per_room * rooms_count + base_price
