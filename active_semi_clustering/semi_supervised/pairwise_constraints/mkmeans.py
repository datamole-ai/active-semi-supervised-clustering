import numpy as np

from sklearn.cluster import KMeans
from metric_learn import MMC


class MKMeans:
    def __init__(self, n_clusters=3, max_iter=1000, diagonal=True):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.diagonal = diagonal

    def fit(self, X, y=None, ml=[], cl=[]):
        X_transformed = X

        if ml and cl:
            # ml_graph, cl_graph, _ = preprocess_constraints(ml, cl, X.shape[0])
            #
            # ml, cl = [], []
            # for i, constraints in ml_graph.items():
            #     for j in constraints:
            #         ml.append((i, j))
            #
            # for i, constraints in cl_graph.items():
            #     for j in constraints:
            #         cl.append((i, j))

            constraints = [np.array(lst) for lst in [*zip(*ml), *zip(*cl)]]
            mmc = MMC(diagonal=self.diagonal)
            mmc.fit(X, constraints=constraints)
            X_transformed = mmc.transform(X)

        kmeans = KMeans(n_clusters=self.n_clusters, init='random', max_iter=self.max_iter)
        kmeans.fit(X_transformed)

        self.labels_ = kmeans.labels_

        return self
