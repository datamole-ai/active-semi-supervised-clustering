import numpy as np

from .helpers import get_constraints_from_neighborhoods
from .example_oracle import MaximumQueriesExceeded


class ExploreConsolidate:
    def __init__(self, n_clusters=3, **kwargs):
        self.n_clusters = n_clusters

    def fit(self, X, oracle=None):
        if oracle.max_queries_cnt <= 0:
            return [], []

        neighborhoods = self._explore(X, self.n_clusters, oracle)
        neighborhoods = self._consolidate(neighborhoods, X, oracle)

        self.pairwise_constraints_ = get_constraints_from_neighborhoods(neighborhoods)

        return self

    def _explore(self, X, k, oracle):
        neighborhoods = []
        traversed = []
        n = X.shape[0]

        x = np.random.choice(n)
        neighborhoods.append([x])
        traversed.append(x)

        try:
            while len(neighborhoods) < k:

                max_distance = 0
                farthest = None

                for i in range(n):
                    if i not in traversed:
                        distance = dist(i, traversed, X)
                        if distance > max_distance:
                            max_distance = distance
                            farthest = i

                new_neighborhood = True
                for neighborhood in neighborhoods:
                    if oracle.query(farthest, neighborhood[0]):
                        neighborhood.append(farthest)
                        new_neighborhood = False
                        break

                if new_neighborhood:
                    neighborhoods.append([farthest])

                traversed.append(farthest)

        except MaximumQueriesExceeded:
            pass

        return neighborhoods

    def _consolidate(self, neighborhoods, X, oracle):
        n = X.shape[0]

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        remaining = set()
        for i in range(n):
            if i not in neighborhoods_union:
                remaining.add(i)

        while True:

            try:
                i = np.random.choice(list(remaining))

                sorted_neighborhoods = sorted(neighborhoods, key=lambda neighborhood: dist(i, neighborhood, X))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(i, neighborhood[0]):
                        neighborhood.append(i)
                        break

                neighborhoods_union.add(i)
                remaining.remove(i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods


def dist(i, S, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
    return distances.min()
