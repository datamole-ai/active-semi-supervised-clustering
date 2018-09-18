import numpy as np
import scipy

from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.farthest_first_traversal import weighted_farthest_first_traversal
from .constraints import preprocess_constraints

np.seterr('raise')


class MPCKMeans:
    "MPCK-Means-S-D that learns only a single (S) diagonal (D) matrix"

    def __init__(self, n_clusters=3, max_iter=10, w=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w

    def fit(self, X, y=None, ml=[], cl=[]):
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize cluster centers
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)

        # Initialize metrics
        A = np.identity(X.shape[1])

        # Repeat until convergence
        for iteration in range(self.max_iter):
            prev_cluster_centers = cluster_centers.copy()

            # Find farthest pair of points according to each metric
            farthest = self._find_farthest_pairs_of_points(X, A)

            # Assign clusters
            labels = self._assign_clusters(X, y, cluster_centers, A, farthest, ml_graph, cl_graph, self.w)

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Update metrics
            A = self._update_metrics(X, labels, cluster_centers, farthest, ml_graph, cl_graph, self.w)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged:
                break

        # print('\t', iteration, converged)

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _find_farthest_pairs_of_points(self, X, A):
        farthest = None
        n = X.shape[0]
        max_distance = 0

        for i in range(n):
            for j in range(n):
                if j < i:
                    distance = self._dist(X[i], X[j], A)
                    if distance > max_distance:
                        max_distance = distance
                        farthest = (i, j, distance)

        assert farthest is not None

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

    def _objective_fn(self, X, i, labels, cluster_centers, cluster_id, A, farthest, ml_graph, cl_graph, w):
        term_d = self._dist(X[i], cluster_centers[cluster_id], A) - np.log(np.linalg.det(A)) / np.log(2)  # FIXME is it okay that it might be negative?

        def f_m(i, j, A):
            return self._dist(X[i], X[j], A)

        def f_c(i, j, A, farthest):
            return farthest[2] - self._dist(X[i], X[j], A)

        term_m = 0
        for j in ml_graph[i]:
            if labels[j] >= 0 and labels[j] != cluster_id:
                term_m += 2 * w * f_m(i, j, A)

        term_c = 0
        for j in cl_graph[i]:
            if labels[j] == cluster_id:
                # assert f_c(i, j, A, farthest) >= 0
                term_c += 2 * w * f_c(i, j, A, farthest)

        return term_d + term_m + term_c

    def _assign_clusters(self, X, y, cluster_centers, A, farthest, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for i in index:
            labels[i] = np.argmin([self._objective_fn(X, i, labels, cluster_centers, cluster_id, A, farthest, ml_graph, cl_graph, w) for cluster_id, cluster_center in enumerate(cluster_centers)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise EmptyClustersException

        return labels

    def _update_metrics(self, X, labels, cluster_centers, farthest, ml_graph, cl_graph, w):
        N, D = X.shape
        A = np.zeros((D, D))

        for d in range(D):
            term_x = np.sum([(x[d] - cluster_centers[labels[i], d]) ** 2 for i, x in enumerate(X)])

            term_m = 0
            for i in range(N):
                for j in ml_graph[i]:
                    if labels[i] != labels[j]:
                        term_m += 1 / 2 * w * (X[i, d] - X[j, d]) ** 2

            term_c = 0
            for i in range(N):
                for j in cl_graph[i]:
                    if labels[i] == labels[j]:
                        tmp = ((X[farthest[0], d] - X[farthest[1], d]) ** 2 - (X[i, d] - X[j, d]) ** 2)
                        term_c += w * max(tmp, 0)

            # print('term_x', term_x, 'term_m', term_m, 'term_c', term_c)

            A[d, d] = N * 1 / max(term_x + term_m + term_c, 1e-9)

        return A

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
