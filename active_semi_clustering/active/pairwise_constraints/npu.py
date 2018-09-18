import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .example_oracle import MaximumQueriesExceeded
from active_semi_clustering.exceptions import EmptyClustersException


class NPU:
    def __init__(self, clusterer=None, **kwargs):
        self.clusterer = clusterer

    def fit(self, X, oracle=None):
        n = X.shape[0]
        ml, cl = [], []
        neighborhoods = []

        x_i = np.random.choice(list(range(n)))
        neighborhoods.append([x_i])

        while True:
            try:
                while True:
                    try:
                        self.clusterer.fit(X, ml=ml, cl=cl)
                    except EmptyClustersException:
                        continue
                    break

                x_i, p_i = self._most_informative(X, self.clusterer, neighborhoods)

                sorted_neighborhoods = list(zip(*reversed(sorted(zip(p_i, neighborhoods)))))[1]
                # print(x_i, neighborhoods, p_i, sorted_neighborhoods)

                must_link_found = False

                for neighborhood in sorted_neighborhoods:

                    must_linked = oracle.query(x_i, neighborhood[0])
                    if must_linked:
                        # TODO is it necessary? this preprocessing is part of the clustering algorithms
                        for x_j in neighborhood:
                            ml.append([x_i, x_j])

                        for other_neighborhood in neighborhoods:
                            if neighborhood != other_neighborhood:
                                for x_j in other_neighborhood:
                                    cl.append([x_i, x_j])

                        neighborhood.append(x_i)
                        must_link_found = True
                        break

                        # TODO should we add the cannot-link in case the algorithm stops before it queries all neighborhoods?

                if not must_link_found:
                    for neighborhood in neighborhoods:
                        for x_j in neighborhood:
                            cl.append([x_i, x_j])

                    neighborhoods.append([x_i])

            except MaximumQueriesExceeded:
                break

        self.pairwise_constraints_ = ml, cl

        return self

    def _most_informative(self, X, clusterer, neighborhoods):
        n = X.shape[0]
        l = len(neighborhoods)

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        unqueried_indices = set(range(n)) - neighborhoods_union

        # TODO if there is only one neighborhood then choose the point randomly?
        if l <= 1:
            return np.random.choice(list(unqueried_indices)), [1]

        # Learn a random forest classifier
        n_estimators = 50
        rf = RandomForestClassifier(n_estimators=n_estimators)
        rf.fit(X, clusterer.labels_)

        # Compute the similarity matrix
        leaf_indices = rf.apply(X)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = (leaf_indices[i,] == leaf_indices[j,]).sum()
        S = S / n_estimators

        p = np.empty((n, l))
        uncertainties = np.zeros(n)
        expected_costs = np.ones(n)

        # For each point that is not in any neighborhood...
        # TODO iterate only unqueried indices
        for x_i in range(n):
            if not x_i in neighborhoods_union:
                for n_i in range(l):
                    p[x_i, n_i] = (S[x_i, neighborhoods[n_i]].sum() / len(neighborhoods[n_i]))

                # If the point is not similar to any neighborhood set equal probabilities of belonging to each neighborhood
                if np.all(p[x_i,] == 0):
                    p[x_i,] = np.ones(l)

                p[x_i,] = p[x_i,] / p[x_i,].sum()

                if not np.any(p[x_i,] == 1):
                    positive_p_i = p[x_i, p[x_i,] > 0]
                    uncertainties[x_i] = -(positive_p_i * np.log2(positive_p_i)).sum()
                    expected_costs[x_i] = (positive_p_i * range(1, len(positive_p_i) + 1)).sum()
                else:
                    uncertainties[x_i] = 0
                    expected_costs[x_i] = 1  # ?

        normalized_uncertainties = uncertainties / expected_costs

        most_informative_i = np.argmax(normalized_uncertainties)
        return most_informative_i, p[most_informative_i]
