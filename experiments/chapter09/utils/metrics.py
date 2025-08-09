# chapter09/utils/metrics.py


from collections import Counter
import math
import numpy as np


def accuracy(groups, precision=3):
    """Взвешенная точность — сколько объектов в группе совпадают с её «модой»."""
    n_instances = sum(len(group) for group in groups)
    total = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        counts_dict = Counter(group)
        most_common_count = max(counts_dict.values())
        total += (most_common_count / len(group)) * (len(group) / n_instances)
    return round(total, precision)


def counts(elements):
    """Считает количество каждого класса в списке."""
    counter = Counter(elements)
    return list(counter.values())


def gini_one_group(elements):
    """Индекс Джини для одной группы элементов."""
    cts = counts(elements)
    n = sum(cts)
    return 1 - sum([p_i ** 2 / n ** 2 for p_i in cts])


def entropy_one_group(elements):
    """Энтропия для одной группы элементов."""
    cts = counts(elements)
    n = sum(cts)
    if n == 0:
        return 0
    props = [c / n for c in cts]
    return -sum(p * math.log2(p) for p in props if p > 0)


def gini_index(groups, precision=3):
    """Взвешенный индекс Джини для нескольких групп."""
    n_instances = sum(len(group) for group in groups)
    total = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        total += gini_one_group(group) * (size / n_instances)
    return round(total, precision)


def entropy(groups, precision=3):
    """Взвешенная энтропия для нескольких групп."""
    n_instances = sum(len(group) for group in groups)
    total = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        total += entropy_one_group(group) * (size / n_instances)
    return round(total, precision)


def mse(arr):
    if len(arr) == 0:
        return 0
    mean = np.mean(arr)
    return np.mean((arr - mean) ** 2)


def mae(arr):
    if len(arr) == 0:
        return 0
    mean = np.mean(arr)
    return np.mean(np.abs(arr - mean))
