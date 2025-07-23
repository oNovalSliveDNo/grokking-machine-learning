# chapter03/utils/reporting


def format_equation(slope, intercept, precision=2):
    """Форматирует уравнение прямой вида y = mx + b"""
    m = round(slope, precision)
    b = round(intercept, precision)
    sign = "+" if b >= 0 else "-"
    return f"y = {m} * x {sign} {abs(b)}"


def print_prediction(slope, intercept, x_value):
    """Форматирует прогноз для заданного x"""
    y = slope * x_value + intercept
    return f"Для {x_value} комнат → Предсказанная цена: {y:.2f}"
