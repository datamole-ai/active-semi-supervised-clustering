import numpy as np

from .kmeans import KMeans


class SeededKMeans(KMeans):
    def _init_cluster_centers(self, X, y=None):
        if np.all(y == -1):
            return X[np.random.choice(X.shape[0], self.n_clusters, replace=False), :]
        else:
            return self._get_cluster_centers(X, y)
