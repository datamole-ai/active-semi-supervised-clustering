import numpy as np
import scipy

from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.farthest_first_traversal import weighted_farthest_first_traversal
from .constraints import preprocess_constraints


# np.seterr('raise')

class MPCKMeansMF:
    """
    MPCK-Means that learns multiple (M) full (F) matrices
    """

    def __init__(self, n_clusters=3, max_iter=100, w=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w

    def fit(self, X, y=None, ml=[], cl=[]):
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize cluster centers
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)

        # Initialize metrics
        As = [np.identity(X.shape[1]) for i in range(self.n_clusters)]

        # Repeat until convergence
        for iteration in range(self.max_iter):
            prev_cluster_centers = cluster_centers.copy()

            # Find farthest pair of points according to each metric
            farthest = self._find_farthest_pairs_of_points(X, As)

            # Assign clusters
            labels = self._assign_clusters(X, y, cluster_centers, As, farthest, ml_graph, cl_graph, self.w)

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Update metrics
            As = self._update_metrics(X, labels, cluster_centers, farthest, ml_graph, cl_graph, self.w)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged:
                break

        # print('\t', iteration, converged)

        self.cluster_centers_, self.labels_ = cluster_centers, labels
        self.As_ = As

        return self

    def _find_farthest_pairs_of_points(self, X, As):
        farthest = [None] * self.n_clusters

        n = X.shape[0]
        for cluster_id in range(self.n_clusters):
            max_distance = 0

            for i in range(n):
                for j in range(n):
                    if j < i:
                        distance = self._dist(X[i], X[j], As[cluster_id])
                        if distance > max_distance:
                            max_distance = distance
                            farthest[cluster_id] = (i, j, distance)

        return farthest

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])
        neighborhood_weights = neighborhood_sizes / neighborhood_sizes.sum()

        # print('\t', len(neighborhoods), neighborhood_sizes)

        if len(neighborhoods) > self.n_clusters:
            cluster_centers = neighborhood_centers[weighted_farthest_first_traversal(neighborhood_centers, neighborhood_weights, self.n_clusters)]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            if len(neighborhoods) < self.n_clusters:
                remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers

    def _dist(self, x, y, A):
        "(x - y)^T A (x - y)"
        return scipy.spatial.distance.mahalanobis(x, y, A) ** 2

    def _objective_function(self, X, i, labels, cluster_centers, cluster_id, As, farthest, ml_graph, cl_graph, w):
        term_d = self._dist(X[i], cluster_centers[cluster_id], As[cluster_id]) - np.log(max(np.linalg.det(As[cluster_id]), 1e-9))

        def f_m(i, c_i, j, c_j, As):
            return 1 / 2 * self._dist(X[i], X[j], As[c_i]) + 1 / 2 * self._dist(X[i], X[j], As[c_j])

        def f_c(i, c_i, j, c_j, As, farthest):
            return farthest[c_i][2] - self._dist(X[i], X[j], As[c_i])

        term_m = 0
        for j in ml_graph[i]:
            if labels[j] >= 0 and labels[j] != cluster_id:
                term_m += 2 * w * f_m(i, cluster_id, j, labels[j], As)

        term_c = 0
        for j in cl_graph[i]:
            if labels[j] == cluster_id:
                term_c += 2 * w * f_c(i, cluster_id, j, labels[j], As, farthest)

        return term_d + term_m + term_c

    def _assign_clusters(self, X, y, cluster_centers, As, farthest, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for i in index:
            labels[i] = np.argmin(
                [self._objective_function(X, i, labels, cluster_centers, cluster_id, As, farthest, ml_graph, cl_graph, w) for cluster_id, cluster_center in enumerate(cluster_centers)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise EmptyClustersException

        return labels

    def _update_metrics(self, X, labels, cluster_centers, farthest, ml_graph, cl_graph, w):
        As = []

        for cluster_id in range(self.n_clusters):
            X_i = X[labels == cluster_id]
            n = X_i.shape[0]

            if n == 1:
                As.append(np.identity(X_i.shape[1]))
                continue

            A_inv = (X_i - cluster_centers[cluster_id]).T @ (X_i - cluster_centers[cluster_id])

            for i in range(X.shape[0]):
                for j in ml_graph[i]:
                    if labels[i] == cluster_id or labels[j] == cluster_id:
                        if labels[i] != labels[j]:
                            A_inv += 1 / 2 * w * ((X[i][:, None] - X[j][:, None]) @ (X[i][:, None] - X[j][:, None]).T)

            for i in range(X.shape[0]):
                for j in cl_graph[i]:
                    if labels[i] == cluster_id or labels[j] == cluster_id:
                        if labels[i] == labels[j]:
                            A_inv += w * (
                                    ((X[farthest[cluster_id][0]][:, None] - X[farthest[cluster_id][1]][:, None]) @ (X[farthest[cluster_id][0]][:, None] - X[farthest[cluster_id][1]][:, None]).T) - (
                                    (X[i][:, None] - X[j][:, None]) @ (X[i][:, None] - X[j][:, None]).T))

            # Handle the case when the matrix is not invertible
            if not self._is_invertible(A_inv):
                # print("Not invertible")
                A_inv += 1e-9 * np.trace(A_inv) * np.identity(A_inv.shape[0])

            A = n * np.linalg.inv(A_inv)

            # Is A positive semidefinite?
            if not np.all(np.linalg.eigvals(A) >= 0):
                # print("Negative definite")
                eigenvalues, eigenvectors = np.linalg.eigh(A)
                A = eigenvectors @ np.diag(np.maximum(0, eigenvalues)) @ np.linalg.inv(eigenvectors)

            As.append(A)

        return As

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def _is_invertible(self, A):
        return A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]
