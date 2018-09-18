import numpy as np


class Random:
    def __init__(self, n_clusters=3, **kwargs):
        self.n_clusters = n_clusters

    def fit(self, X, oracle=None):
        constraints = [np.random.choice(range(X.shape[0]), size=2, replace=False).tolist() for _ in range(oracle.max_queries_cnt)]

        ml, cl = [], []

        for i, j in constraints:
            must_linked = oracle.query(i, j)
            if must_linked:
                ml.append((i, j))
            else:
                cl.append((i, j))

        self.pairwise_constraints_ = ml, cl

        return self
