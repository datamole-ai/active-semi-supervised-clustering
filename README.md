# active-semi-supervised-clustering

Active semi-supervised clustering algorithms for scikit-learn.

## Algorithms

### Semi-supervised clustering

* Seeded-KMeans
* Constrainted-KMeans
* COP-KMeans
* Pairwise constrained K-Means (PCK-Means)
* Metric K-Means (MK-Means)
* Metric pairwise constrained K-Means (MPCK-Means)

### Active learning of pairwise clustering

* Explore & Consolidate
* Min-max
* Normalized point-based uncertainty (NPU) method

## Installation

```
pip install active-semi-supervised-clustering
```

## Usage

```python
from sklearn import datasets, metrics
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax
```

```python
X, y = datasets.load_iris(return_X_y=True)
```

First, obtain some pairwise constraints from an oracle.

```python
# TODO implement your own oracle that will, for example, query a domain expert via GUI or CLI
oracle = ExampleOracle(y, max_queries_cnt=10)

active_learner = MinMax(n_clusters=3)
active_learner.fit(X, oracle=oracle)
pairwise_constraints = active_learner.pairwise_constraints_
```

Then, use the constraints to do the clustering.

```python
clusterer = PCKMeans(n_clusters=3)
clusterer.fit(X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
```

Evaluate the clustering using Adjusted Rand Score.

```python
metrics.adjusted_rand_score(y, clusterer.labels_)
```
