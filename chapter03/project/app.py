import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models.linear_regression import linear_regression, predict

# Заголовок приложения
st.title("📈 Интерактивная линейная регрессия")

# Генерация синтетических данных
np.random.seed(42)
rooms = np.random.randint(1, 6, 20)
prices = 50 * rooms + 100 + np.random.normal(0, 20, 20)

# Настройки модели в сайдбаре
with st.sidebar:
    st.header("Настройки модели")
    trick = st.selectbox("Метод обновления весов", ["Simple", "Absolute", "Square"])
    error = st.selectbox("Функция ошибки", ["MAE", "MSE", "RMSE"])
    mode = st.selectbox("Режим обучения", ["SGD", "Batch", "Mini"])
    learning_rate = st.slider("Скорость обучения (η)", 0.001, 0.1, 0.01)
    epochs = st.slider("Количество эпох", 100, 5000, 1000)

# Обучение модели
price_per_room, base_price, errors = linear_regression(
    rooms, prices,
    trick=str(trick).lower(),
    error=str(error).lower(),
    mode=str(mode).lower(),
    learning_rate=learning_rate,
    epochs=epochs
)

# Предсказание
st.subheader("Предсказание цены")
rooms_input = st.number_input("Введите количество комнат", min_value=1, max_value=10, value=3)
predicted_price = predict(price_per_room, base_price, rooms_input)
st.write(f"Предсказанная цена: **${predicted_price:.2f}**")

# Вывод параметров модели
st.subheader("Параметры модели")
st.write(f"- Цена за комнату: `{price_per_room:.2f}`")
st.write(f"- Базовая цена: `{base_price:.2f}`")

# Создаем две колонки для графиков
col1, col2 = st.columns(2)

# График данных и линии регрессии в первой колонке
with col1:
    st.subheader("Модель и данные")

    # Создаем график Plotly
    fig = go.Figure()

    # Добавляем реальные данные (точки)
    fig.add_trace(go.Scatter(
        x=rooms,
        y=prices,
        mode='markers',
        name='Реальные данные',
        marker=dict(color='blue', size=8)
    ))

    # Добавляем линию регрессии
    x_range = np.linspace(1, 10, 100)
    y_pred = predict(price_per_room, base_price, x_range)
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        name='Модель',
        line=dict(color='red', width=2)
    ))

    # Настраиваем оформление
    fig.update_layout(
        xaxis_title="Количество комнат",
        yaxis_title="Цена ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# График ошибки во второй колонке
with col2:
    st.subheader("График ошибки")

    # Создаем график ошибки
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=np.arange(len(errors)),
        y=errors,
        mode='lines',
        name='Ошибка',
        line=dict(color='green', width=2)
    ))

    # Настраиваем оформление
    fig2.update_layout(
        xaxis_title="Эпоха",
        yaxis_title=error,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig2, use_container_width=True)
