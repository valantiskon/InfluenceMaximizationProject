'''
Implements degreeDiscount heuristic that stops after necessary amount of nodes is targeted.
Now it calculates spread of influence after a step increase in seed size
and returns if targeted set size is greater than desired input tsize.
'''

import math
import os.path
import time

import networkx as nx

# Import degree heuristics
from degreeDiscount import degreeDiscountIC
from degreeHeuristic import degreeHeuristic

# Import independent cascade model
from IC import runIC


def binarySearchBoundary(G, k, Tsize, targeted_size, step, p, iterations, degreeDiscount_Heuristic='degreeDiscount', initial_time=0):
    # Calculate the time it takes to select each node, initial_time is added in order keep track of the time needed for
    # the selection of initial nodes
    start_time = time.time()
    # keep a list for time needed to select each nodes
    timer_each_node = []
    timer_each_node.append(initial_time)

    # initialization for binary search
    R = iterations
    stepk = -int(math.ceil(float(step)/2))
    k += stepk
    if k not in Tsize:

        if (degreeDiscount_Heuristic == 'degreeDiscount'):
            S = degreeDiscountIC(G, k, p)
        else:
            S = degreeHeuristic(G, k, p)

        avg = 0
        for i in range(R):
            T = runIC(G, S, p)
            avg += float(len(T))/R


        timer_each_node.append(time.time() + initial_time - start_time)


        Tsize[k] = avg


    # check values of Tsize in between last 2 calculated steps
    while stepk != 1:
        print(k, stepk, Tsize[k])
        if Tsize[k] >= targeted_size:
            stepk = -int(math.ceil(float(abs(stepk))/2))
        else:
            stepk = int(math.ceil(float(abs(stepk))/2))
        k += stepk

        if k not in Tsize:

            if (degreeDiscount_Heuristic == 'degreeDiscount'):
                S = degreeDiscountIC(G, k, p)
            else:
                S = degreeHeuristic(G, k, p)


            # stores the influence spread of each node
            influence_spread = []


            avg = 0
            for i in range(R):
                T = runIC(G, S, p)
                avg += float(len(T))/R

                influence_spread.append(avg)

            # keep time for each NEW node selected
            timer_each_node.append(time.time() + initial_time - start_time)


            Tsize[k] = avg
            print("datafaq: ", Tsize)
            print("datafaq[k]: ", Tsize[k])

    print("leeeen: ", len(timer_each_node))
    return S, influence_spread, timer_each_node
    #return S, Tsize

def spreadDegreeDiscount(G, targeted_size, step=1, p=.01, iterations=200, degreeDiscount_Heuristic='degreeDiscount'):
    ''' Finds initial set of nodes to propagate in Independent Cascade model
    Input: G -- networkx graph object
    targeted_size -- desired size of targeted set
    step -- step after each to calculate spread
    p -- propagation probability
    R -- number of iterations to average influence spread
    Output:
    S -- seed set that achieves targeted_size
    Tsize -- averaged targeted size for different sizes of seed set
    '''

    # calculate the time  of selecting the first initial nodes (afterwards we will select some of them)
    start_time = time.time()


    Tsize = dict()
    k = 0
    Tsize[k] = 0
    R = iterations

    while Tsize[k] <= targeted_size:
        k += step

        if (degreeDiscount_Heuristic == 'degreeDiscount'):
            S = degreeDiscountIC(G, k, p)
        else:
            S = degreeHeuristic(G, k, p)


        avg = 0
        for i in range(R):
            T = runIC(G, S, p)
            avg += float(len(T))/R
        Tsize[k] = avg

        print(k, Tsize[k])

    # binary search for optimal solution
    return binarySearchBoundary(G, k, Tsize, targeted_size, step, p, iterations, initial_time=time.time()-start_time)

if __name__ == '__main__':
    import time
    start = time.time()

    direct = os.getcwd()  # Gets the current working directory
    # '\\id_obama.csv'

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

    targeted_size = 100
    S, Tsize = spreadDegreeDiscount(G, targeted_size, step=50) # step = 100 -> targeted_size = 200
    print(time.time() - start)

    console = []