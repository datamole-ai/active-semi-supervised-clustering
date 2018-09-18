import numpy as np

from active_semi_clustering.exceptions import EmptyClustersException


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None, **kwargs):
        # Initialize cluster centers
        cluster_centers = self._init_cluster_centers(X, y)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            prev_cluster_centers = cluster_centers.copy()

            # Assign clusters
            labels = self._assign_clusters(X, y, cluster_centers, self._dist)

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _init_cluster_centers(self, X, y=None):
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]

    def _dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def _assign_clusters(self, X, y, cluster_centers, dist):
        labels = np.full(X.shape[0], fill_value=-1)

        for i, x in enumerate(X):
            labels[i] = np.argmin([dist(x, c) for c in cluster_centers])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            raise EmptyClustersException

        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
