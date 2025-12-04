import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

X = np.loadtxt('data_clustering.txt', delimiter=',')

bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Кластеризація даних методом зсуву середнього
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Витягування центрів кластерів
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

plt.figure()
markers = 'o*xvs'
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

for i in range(num_clusters):
    # Відображення точок поточного кластера
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=markers[i % len(markers)], color=colors[i % len(colors)])

    # Відображення центру кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
             markerfacecolor='k', markeredgecolor='k', markersize=15)

plt.title('Кластери (Mean Shift)')
plt.show()
