# app.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apputil import *

sns.set_theme(style="whitegrid")

print("\n--- Exercise 1 ---")
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
centroids, labels = kmeans(X, k=3)
print("Centroids:\n", centroids)
print("Labels:", labels)
print("centroids[1]:", centroids[1])
print("labels[2]:", labels[2])

print("\n--- Exercise 2 ---")
centroids_d, labels_d = kmeans_diamonds(n=1000, k=5)
print("Diamonds centroids shape:", centroids_d.shape)
print("Example centroid (cluster 4):", centroids_d[3])
print("Example label (diamond 10):", labels_d[9])

print("\n--- Exercise 3 --- (Timing)")
n_values = np.arange(100, 5000, 1000)
k5_times = [kmeans_timer(n, 5, 5) for n in n_values]
k_values = np.arange(2, 20)
n10k_times = [kmeans_timer(1000, k, 3) for k in k_values]

# Plot time complexity
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
fig.tight_layout()
fig.suptitle("KMeans Time Complexity", y=1.08, fontsize=14)

sns.lineplot(x=n_values, y=k5_times, ax=axes[0])
axes[0].set_xlabel("Number of Rows (n)")
axes[0].set_ylabel("Time (seconds)")
axes[0].set_title('Increasing n for k=5 Clusters')

sns.lineplot(x=k_values, y=n10k_times, ax=axes[1])
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_title('Increasing k for n=1000 Samples')

plt.show()

print("\n--- Bonus: Binary Search Step Count ---")
step_counts = []
n_vals = np.arange(10, 10000, 500)
for n in n_vals:
    bin_search(n)
    step_counts.append(step_count)

plt.figure(figsize=(6,4))
plt.plot(n_vals, step_counts, marker='o')
plt.title("Binary Search Step Count Growth")
plt.xlabel("Array Size (n)")
plt.ylabel("Step Count")
plt.grid(True)
plt.show()

print("Asymptotic complexity: O(log n)")