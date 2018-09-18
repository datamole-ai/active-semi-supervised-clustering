from active_semi_clustering.exceptions import InconsistentConstraintsException


# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def preprocess_constraints(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # Run DFS from each node to get all the graph's components
    # and add an edge for each pair of nodes in the component (create a complete graph)
    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    visited = [False] * n
    neighborhoods = []
    for i in range(n):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in cl:
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise InconsistentConstraintsException('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods
