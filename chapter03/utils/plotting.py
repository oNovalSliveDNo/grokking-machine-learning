# chapter03/utils/plotting


import numpy as np
import matplotlib.pyplot as plt


def draw_line(slope, intercept, color='red', linewidth=0.7, start=0, end=8):
    x = np.linspace(start, end, 1000)
    y = slope * x + intercept
    plt.plot(x, y, linestyle='-', color=color, linewidth=linewidth)


def plot_points(features, labels):
    plt.scatter(features, labels, color='blue')
    plt.xlabel('Количество комнат')
    plt.ylabel('Цена')
    plt.grid(True, linestyle='--', alpha=0.7)


def plot_errors(errors_list, error_type):
    # График ошибки
    plt.scatter(range(len(errors_list)), errors_list, s=8)
    plt.title(f"Error Function ({error_type.upper()})")
    plt.xlabel("Epoch")
    plt.ylabel(error_type.upper())
    plt.grid(True)
    plt.show()
