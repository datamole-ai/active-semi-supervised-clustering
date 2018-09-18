class MaximumQueriesExceeded(Exception):
    pass


class ExampleOracle:
    def __init__(self, labels, max_queries_cnt=20):
        self.labels = labels
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt

    def query(self, i, j):
        "Query the oracle to find out whether i and j should be must-linked"
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            return self.labels[i] == self.labels[j]
        else:
            raise MaximumQueriesExceeded
