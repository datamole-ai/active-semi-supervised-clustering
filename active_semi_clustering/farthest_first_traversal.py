import numpy as np


def dist(i, S, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
    return distances.min()


def farthest_first_traversal(points, k):
    traversed = []

    # Choose the first point randomly
    i = np.random.choice(len(points))
    traversed.append(i)

    # Find remaining n - 1 maximally separated points
    for _ in range(k - 1):
        max_dst, max_dst_index = 0, None

        for i in range(len(points)):
            if i not in traversed:
                dst = dist(i, traversed, points)

                if dst > max_dst:
                    max_dst = dst
                    max_dst_index = i

        traversed.append(max_dst_index)

    return traversed


def weighted_farthest_first_traversal(points, weights, k):
    traversed = []

    # Choose the first point randomly (weighted)
    i = np.random.choice(len(points), size=1, p=weights)[0]
    traversed.append(i)

    # Find remaining n - 1 maximally separated points
    for _ in range(k - 1):
        max_dst, max_dst_index = 0, None

        for i in range(len(points)):
            if i not in traversed:
                dst = dist(i, traversed, points)
                weighted_dst = weights[i] * dst

                if weighted_dst > max_dst:
                    max_dst = weighted_dst
                    max_dst_index = i

        traversed.append(max_dst_index)

    return traversed
