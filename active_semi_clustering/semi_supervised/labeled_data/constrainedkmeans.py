import numpy as np

from .kmeans import EmptyClustersException
from .seededkmeans import SeededKMeans


class ConstrainedKMeans(SeededKMeans):
    def _assign_clusters(self, X, y, cluster_centers, dist):
        labels = np.full(X.shape[0], fill_value=-1)

        for i, x in enumerate(X):
            if y[i] != -1:
                labels[i] = y[i]
            else:
                labels[i] = np.argmin([dist(x, c) for c in cluster_centers])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            raise EmptyClustersException

        return labels
