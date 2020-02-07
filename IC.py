''' Independent cascade model for influence propagation
'''


def runIC(G, S, p=.01):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial set of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    from copy import deepcopy
    from random import random
    T = deepcopy(S)  # copy already selected nodes

    # ugly C++ version
    i = 0
    while i < len(T):
        for v in G[T[i]]:  # for neighbors of a selected node
            if v not in T:  # if it wasn't selected yet
                w = G[T[i]][v]['weight']  # count the number of edges between two nodes
                if random() <= 1 - (1 - p) ** w:  # if at least one of edges propagate influence
#                    print(T[i], 'influences', v)
                    T.append(v)
        i += 1

    return T


# calculates the influence spread by calculating the average nodes affected each time we choose a new node
def avgSize(G, S, p, iterations):
    avg = 0
    for i in range(iterations):
        avg += float(len(runIC(G, S, p))) / iterations
    return avg