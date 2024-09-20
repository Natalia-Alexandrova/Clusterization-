import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


def get_size(data: pd.DataFrame, labels: pd.Series, ax=None) -> None:
    """
    Фунция для вывода графика размера кластеров

    data - датасет эмбеддингов
    labels - метки кластеров
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(7, 5)

    cluster_size = data.assign(cluster=labels).groupby(
        'cluster').count().iloc[:, 0]
    ax = cluster_size.sort_values().plot.barh(figsize=(6, 4), color='C0')
    plt.title('Размер кластеров')

    return


def plot(data: pd.DataFrame, labels: pd.Series, ax=None):
    """
    Фунция отрисовки кластеров в 3D

    data - датасет эмбеддингов
    labels - метки кластеров
    """

    if ax is None:
        ax = plt.figure(figsize=(10, 7)).gca(projection='3d')

    data = data.to_numpy()

    ax.scatter(
        xs=data[:, 0],
        ys=data[:, 1],
        zs=data[:, 2],
        c=labels)

    ax.set_title(f'Визуализация ({len(np.unique(labels))}) кластеров', y=1.02)
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.set_zlabel('UMAP-3')

    return


def silhouette_plot(data: pd.DataFrame, labels: pd.Series, metrics='euclidean', ax=None) -> None:
    """
    Функция вывода графика силуэтного скора

    data - датасет эмбеддингов
    labels - метки кластеров
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(7, 5)

    silhouette_vals = silhouette_samples(data, labels, metric=metrics)

    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.barh(range(y_lower, y_upper), cluster_silhouette_vals,
                edgecolor='none', height=1)
        ax.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Получение средней оценки силуэтного скора и построение графика
    avg_score = np.mean(silhouette_vals)
    ax.axvline(avg_score, linestyle='--', linewidth=1, color='red')
    ax.set_xlabel(f'Silhouette  = {round(avg_score, 3)}')
    ax.set_ylabel('Метки кластеров')
    ax.set_title(f'График силуэта для различных кластеров ({len(np.unique(labels))})', y=1.02)

    return
