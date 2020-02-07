import math
import os.path
import time

import networkx as nx

# Import degree heuristics
from degreeDiscount import degreeDiscountIC
from degreeHeuristic import degreeHeuristic

# Import independent cascade model
from IC import *


def binaryDegreeDiscount (G, tsize, p=.01, a=0.38, step=5, iterations=200, degreeDiscount_Heuristic='degreeDiscount'):
    ''' Finds minimal number of nodes necessary to reach tsize number of nodes
    using degreeDiscount algorithms and binary search.
    Input: G -- networkx graph object
    tsize -- number of nodes necessary to reach
    p -- propagation probability
    a -- fraction of tsize to use as initial seed set size
    step -- step between iterations of binary search
    iterations -- number of iterations to average independent cascade
    degreeDiscount_Heuristic -- whether to select degree Discount or degree Heuristic algorithm
    Output:
    S -- seed set
    Tspread -- spread values for different sizes of seed set
    '''

    # Calculate the time it takes to select each node
    start_time = time.time()
    # keep a list for time needed to select each nodes
    timer_each_node = []

    Tspread = dict()
    # find initial total spread
    k0 = int(a*tsize)

    if(degreeDiscount_Heuristic == 'degreeDiscount'):
        S = degreeDiscountIC(G, k0, p)
    else:
        S = degreeHeuristic(G, k0, p)


    t = avgSize(G, S, p, iterations)
    Tspread[k0] = t
    # find bound (lower or upper) of total spread
    k = k0
    print(k, step, Tspread[k])

    # keep time for each NEW node selected
    timer_each_node.append(time.time() - start_time)

    if t >= tsize:
        # find the value of k that doesn't spread influence up to tsize nodes
        step *= -1
        while t >= tsize:
            # reduce step if necessary
            while k + step < 0:
                step = int(math.ceil(float(step)/2))
            k += step

            if (degreeDiscount_Heuristic == 'degreeDiscount'):
                S = degreeDiscountIC(G, k, p)
            else:
                S = degreeHeuristic(G, k, p)

            t = avgSize(G, S, p, iterations)
            Tspread[k] = t
            print(k, step, Tspread[k])

            # keep time for each NEW node selected
            timer_each_node.append(time.time() - start_time)
    else:
        # find the value of k that spreads influence up to tsize nodes
        while t < tsize:
            k += step

            if (degreeDiscount_Heuristic == 'degreeDiscount'):
                S = degreeDiscountIC(G, k, p)
            else:
                S = degreeHeuristic(G, k, p)

            t = avgSize(G, S, p, iterations)
            Tspread[k] = t
            print(k, step, Tspread[k])

            # keep time for each NEW node selected
            timer_each_node.append(time.time() - start_time)

    if Tspread[k] < Tspread[k-step]:
        k -= step
        step = abs(step)

    # search precise boundary
    stepk = step
    while abs(stepk) != 1:
        if Tspread[k] >= tsize:
            stepk = -int(math.ceil(float(abs(stepk))/2))
        else:
            stepk = int(math.ceil(float(abs(stepk))/2))
        k += stepk

        if k not in Tspread:

            if (degreeDiscount_Heuristic == 'degreeDiscount'):
                S = degreeDiscountIC(G, k, p)
            else:
                S = degreeHeuristic(G, k, p)

            Tspread[k] = avgSize(G, S, p, iterations)

            # keep time for each NEW node selected
            timer_each_node.append(time.time() - start_time)
        print(k, stepk, Tspread[k])

        print("(number of nodes) - (spread) : ", Tspread)

# ======================================================================================================================
        # stores the influence spread of each node
        influence_spread = []
        avg = 0
        for i in range(iterations):
            T = runIC(G, S, p)
            avg += float(len(T)) / iterations
            influence_spread.append(avg)

        print("(spread) : ", influence_spread)

    return S, influence_spread, timer_each_node
    #return S, Tspread

if __name__ == '__main__':
    import time
    start = time.time()

    direct = os.getcwd()  # Gets the current working directory
    #'\\id_obama.csv'

    # read in graph
    G = nx.Graph()
    with open(direct + "\\ca-CSphd.txt") as f:
        n, m = f.readline().split()
        for line in f:
            u, v = map(int, line.split())
            try:
                G[u][v]['weight'] += 1
            except:
                G.add_edge(u, v, weight=1)
    print('Built graph G')
    print(time.time() - start)

    tsize = 100
    S, Tsize = binaryDegreeDiscount(G, tsize, step=5)
    print('Necessary %s initial nodes to target %s nodes in graph G' %(len(S), tsize))
    print(S)
    print(time.time() - start)
    console = []