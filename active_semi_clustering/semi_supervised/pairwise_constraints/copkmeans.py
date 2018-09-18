import numpy as np

from active_semi_clustering.exceptions import EmptyClustersException, ClusteringNotFoundException
from .constraints import preprocess_constraints


class COPKMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None, ml=[], cl=[]):
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize cluster centers
        cluster_centers = self._init_cluster_centers(X)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            prev_cluster_centers = cluster_centers.copy()

            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, self._dist, ml_graph, cl_graph)

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _init_cluster_centers(self, X):
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]

    def _dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def _assign_clusters(self, *args):
        max_retries_cnt = 1000

        for retries_cnt in range(max_retries_cnt):
            try:
                return self._try_assign_clusters(*args)

            except ClusteringNotFoundException:
                continue

        raise ClusteringNotFoundException

    def _try_assign_clusters(self, X, cluster_centers, dist, ml_graph, cl_graph):
        labels = np.full(X.shape[0], fill_value=-1)

        data_indices = list(range(X.shape[0]))
        np.random.shuffle(data_indices)

        for i in data_indices:
            distances = np.array([dist(X[i], c) for c in cluster_centers])
            # sorted_cluster_indices = np.argsort([dist(x, c) for c in cluster_centers])

            for cluster_index in distances.argsort():
                if not self._violates_constraints(i, cluster_index, labels, ml_graph, cl_graph):
                    labels[i] = cluster_index
                    break

            if labels[i] < 0:
                raise ClusteringNotFoundException

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            raise EmptyClustersException

        return labels

    def _violates_constraints(self, i, cluster_index, labels, ml_graph, cl_graph):
        for j in ml_graph[i]:
            if labels[j] > 0 and cluster_index != labels[j]:
                return True

        for j in cl_graph[i]:
            if cluster_index == labels[j]:
                return True

        return False

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
