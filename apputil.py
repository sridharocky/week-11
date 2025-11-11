# apputil.py

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from time import time

# ---------------------------
# Exercise 1
# ---------------------------
def kmeans(X, k):
    """
    Performs k-means clustering on numerical NumPy array X.
    Returns (centroids, labels).
    """
    model = KMeans(n_clusters=k, n_init='auto', random_state=42)
    model.fit(X)
    centroids = model.cluster_centers_
    labels = model.labels_
    return centroids, labels


# ---------------------------
# Exercise 2
# ---------------------------
# Load diamonds dataset and keep only numeric columns
diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include=[np.number])
# Confirm this has 7 numerical columns: carat, depth, table, price, x, y, z

def kmeans_diamonds(n, k):
    """
    Runs kmeans() on first n rows of the numeric diamonds dataset.
    Returns (centroids, labels)
    """
    X = diamonds_numeric.head(n).to_numpy()
    centroids, labels = kmeans(X, k)
    return centroids, labels


# ---------------------------
# Exercise 3
# ---------------------------
def kmeans_timer(n, k, n_iter=5):
    """
    Runs kmeans_diamonds(n, k) n_iter times, returning average runtime in seconds.
    """
    times = []
    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        times.append(time() - start)
    return np.mean(times)


# ---------------------------
# Bonus Exercise
# ---------------------------
step_count = 0

def bin_search(n):
    """
    Binary search counting computational steps.
    Returns index if found, -1 otherwise.
    """
    global step_count
    step_count = 0

    arr = np.arange(n)
    left = 0
    right = n - 1
    x = n - 1

    while left <= right:
        step_count += 1  # count loop iteration
        middle = left + (right - left) // 2
        step_count += 1  # for computing middle

        if arr[middle] == x:
            step_count += 1  # for comparison
            return middle
        elif arr[middle] < x:
            step_count += 1
            left = middle + 1
        else:
            step_count += 1
            right = middle - 1
    return -1
