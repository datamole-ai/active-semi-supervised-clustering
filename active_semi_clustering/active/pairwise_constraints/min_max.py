import numpy as np

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate


class MinMax(ExploreConsolidate):
    def _consolidate(self, neighborhoods, X, oracle):
        n = X.shape[0]

        skeleton = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                skeleton.add(i)

        remaining = set()
        for i in range(n):
            if i not in skeleton:
                remaining.add(i)

        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.sqrt(((X[i] - X[j]) ** 2).sum())

        kernel_width = np.percentile(distances, 20)

        while True:
            try:
                max_similarities = np.full(n, fill_value=float('+inf'))
                for x_i in remaining:
                    max_similarities[x_i] = np.max([similarity(X[x_i], X[x_j], kernel_width) for x_j in skeleton])

                q_i = max_similarities.argmin()

                sorted_neighborhoods = reversed(sorted(neighborhoods, key=lambda neighborhood: np.max([similarity(X[q_i], X[n_i], kernel_width) for n_i in neighborhood])))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(q_i, neighborhood[0]):
                        neighborhood.append(q_i)
                        break

                skeleton.add(q_i)
                remaining.remove(q_i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods


def similarity(x, y, kernel_width):
    return np.exp(-((x - y) ** 2).sum() / (2 * (kernel_width ** 2)))
