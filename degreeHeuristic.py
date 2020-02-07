''' Implementation of degree heuristic[1] for Independent Cascade model
of influence propagation in graph G.
Takes k nodes with the largest degree.
[1] -- Wei Chen et al. Efficient influence maximization in Social Networks
'''


def degreeHeuristic(G, k, p=.01):
    ''' Finds initial set of nodes to propagate in Independent Cascade model
    Input: G -- networkx graph object
    k -- number of nodes needed
    p -- propagation probability
    Output:
    S -- chosen k nodes
    '''
    S = []
    d = dict()
    for u in G:
        degree = sum([G[u][v]['weight'] for v in G[u]])
        # degree = len(G[u])
        d[u] = degree
    for i in range(k):
        # dd saves tuples, max function of a tuple compares the first value in the  tuple, if it the same then compare the second,
        # we want to compare only the second, so x being a tuple with x[1] we select the second value of the tuple
        u, degree = max(d.items(), key=lambda x: x[1])
        #u, degree = max(d.iteritems(), key=lambda (k,v): v)
        d.pop(u)
        S.append(u)
    return S