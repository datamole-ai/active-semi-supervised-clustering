import numpy as np

from sklearn.cluster import KMeans
from metric_learn import RCA

from .constraints import preprocess_constraints


class RCAKMeans:
    """
    Relative Components Analysis (RCA) + KMeans
    """

    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, y=None, ml=[], cl=[]):
        X_transformed = X

        if ml:
            chunks = np.full(X.shape[0], -1)
            ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])
            for i, neighborhood in enumerate(neighborhoods):
                chunks[neighborhood] = i

            # print(chunks)

            rca = RCA()
            rca.fit(X, chunks=chunks)
            X_transformed = rca.transform(X)

            # print(rca.metric())

        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)
        kmeans.fit(X_transformed)

        self.labels_ = kmeans.labels_

        return self
