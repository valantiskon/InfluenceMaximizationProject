''' Implementation of degree discount heuristic [1] for Independent Cascade model
of influence propagation in graph G
[1] -- Wei Chen et al. Efficient influence maximization in Social Networks (algorithm 4)
'''


def degreeDiscountIC(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model (without priority queue)
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    Note: the routine runs twice slower than using PQ. Implemented to verify results
    '''
    d = dict()
    dd = dict()  # degree discount
    t = dict()  # number of selected neighbors
    S = []  # selected set of nodes
    for u in G:
        d[u] = sum([G[u][v]['weight'] for v in G[u]])  # each edge adds degree 1
        # d[u] = len(G[u]) # each neighbor adds degree 1
        dd[u] = d[u]
        t[u] = 0
    for i in range(k):
        # dd saves tuples, max function of a tuple compares the first value in the  tuple, if it the same then compare the second,
        # we want to compare only the second, so x being a tuple with x[1] we select the second value of the tuple
        u, ddv = max(dd.items(), key=lambda x: x[1])
#        u, ddv = max(dd.items(), key=lambda (k,v): v)
        dd.pop(u)
        S.append(u)
        for v in G[u]:
            if v not in S:
                t[v] += G[u][v]['weight']  # increase number of selected neighbors
                dd[v] = d[v] - 2 * t[v] - (d[v] - t[v]) * t[v] * p
    return S

'''
def single_degree_discount(graph, k):
    degree_count = dict(graph.degree)
    topk = []
    neighborhood_fn = graph.neighbors # if isinstance(graph, nx.Graph) else graph.predecessors
    for _ in range(k):
        node = max(degree_count.items(), key=lambda x: x[1])[0]
        topk.append(node)
        for neighbor in neighborhood_fn(node):
            degree_count[neighbor] -= 1
    return topk
'''

if __name__ == '__main__':
    console = []