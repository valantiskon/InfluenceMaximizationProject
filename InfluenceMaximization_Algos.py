# Import packages
import matplotlib.pyplot as plt
from random import uniform, seed
import numpy as np
import time
import networkx as nx
import pandas as pd
import os.path
from binaryDegreeDiscount import *
from spreadDegreeDiscount import *

# Calculate the influence_spread using Independent Cascade model

# graph:  networkx graph
# influential_nodes: set of seed nodes
# propagation_proba: propagation probability
# monte_carlo_sim: the number of Monte-Carlo simulations
def IndependentCascadeModel(graph, influential_nodes, propagation_proba=0.5, monte_carlo_sim=1000):
    # implement Monte-Carlo Simulations
    influence_spread = []
    for i in range(monte_carlo_sim):
        # influence influence_spread calculation
        new_active_node = influential_nodes[:]
        activated_nodes_list = influential_nodes[:]

        while new_active_node:
            # find neighbors of new active nodes that activate
            new_nodes = []
            for node in new_active_node:
                np.random.seed(i)
                select_new_effected_nodes = np.random.uniform(0, 1, len(list(graph.successors(node)))) < propagation_proba
                #select_new_effected_nodes = effected_nodes < propagation_proba
                new_nodes += (list(np.extract(select_new_effected_nodes, list(graph.successors(node)))))

            new_active_node = list(set(new_nodes) - set(activated_nodes_list))
            activated_nodes_list += new_active_node

        influence_spread.append(len(activated_nodes_list))

    # the average number of nodes influenced by the selected, possible most influential nodes simulates the influence influence_spread
    return np.mean(influence_spread)



# Greedy Algorithm
# graph:  networkx graph
# number_influential_nodes: number of influential nodes to find
# propagation_proba: propagation probability
# monte_carlo_sim: the number of Monte-Carlo simulations
def greedy_influence_maximization(graph, number_influential_nodes, propagation_proba=0.1, monte_carlo_sim=1000):
    influential_nodes = []
    influence_spread = []
    time_per_node = []
    start_time = time.time()

    # Find nodes with largest influence spread
    for _ in range(number_influential_nodes):
        # check the nodes not in the list of influential nodes
        best_spread = 0
        for a_node in set(range(len(list(graph.nodes())))) - set(influential_nodes):
            # Get the influence spread
            inf_spread_new_node = IndependentCascadeModel(graph, influential_nodes + [a_node], propagation_proba, monte_carlo_sim)
            # Update the winning node and influence_spread so far
            if inf_spread_new_node > best_spread:
                best_spread = inf_spread_new_node
                node = a_node
        influential_nodes.append(node)
        influence_spread.append(best_spread)
        time_per_node.append(time.time() - start_time)

    # return most influential nodes, total influence_spread, time for each calculation
    return influential_nodes, influence_spread, time_per_node




# Cost Effective Lazy Forward (CELF) Algorithm
# graph:  networkx graph
# number_influential_nodes: number of influential nodes to find
# propagation_proba: propagation probability
# monte_carlo_sim: the number of Monte-Carlo simulations
def celf_algorithm(graph, number_influential_nodes, propagation_proba=0.1, monte_carlo_sim=1000):
    # Find the first node with greedy algorithm
    start_time = time.time()
    influence_spread_per_node = [IndependentCascadeModel(graph, [graph_node], propagation_proba, monte_carlo_sim) for graph_node in graph.nodes()]
    # save in a list sorted of nodes by their influence spread
    sorted_nodes_infl = sorted(zip(list(graph.nodes()), influence_spread_per_node), key=lambda x: x[1], reverse=True)

    # select the most influential node
    influential_nodes = [sorted_nodes_infl[0][0]]
    influence_spread = sorted_nodes_infl[0][1]
    spread_all_nodes = [sorted_nodes_infl[0][1]]
    sorted_nodes_infl = sorted_nodes_infl[1:]
    time_per_node = [time.time() - start_time]

    for _ in range(number_influential_nodes-1):
        change_of_top_node = False
        while not change_of_top_node:
            # Recalculate influence_spread of top node
            current = sorted_nodes_infl[0][0]
            # Evaluate the influence_spread function and store the marginal gain in the list
            sorted_nodes_infl[0] = (current, IndependentCascadeModel(graph, influential_nodes + [current], propagation_proba, monte_carlo_sim) - influence_spread)
            # Re-sort the list
            sorted_nodes_infl = sorted(sorted_nodes_infl, key=lambda value: value[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            if(sorted_nodes_infl[0][0] == current):
                change_of_top_node = 1
            else:
                change_of_top_node = 0

        influence_spread += sorted_nodes_infl[0][1]
        influential_nodes.append(sorted_nodes_infl[0][0])
        spread_all_nodes.append(influence_spread)
        time_per_node.append(time.time() - start_time)

        # remove the selected node from the list
        sorted_nodes_infl = sorted_nodes_infl[1:]

    # return most influential nodes, total influence_spread, time for each calculation
    return influential_nodes, spread_all_nodes, time_per_node





# ======================================================================================================================
# START
# ======================================================================================================================


# Generate Graph
#F = open('re_obama.txt', 'r')
#m = np.matrix(F)
direct = os.getcwd()  # Gets the current working directory
#file_A = direct + '\\id_obama.csv'

'''
# EDIT FILES

train_A = pd.read_csv(file_A)
# Drop the first column of reading file
train_A.drop(["numb"], axis=1, inplace=True)
#train_A.to_csv(direct + '\\id_obama.csv', index=False)
print(train_A)
'''

# ======================================================================================================================
# Artifical dataset
# ======================================================================================================================
'''
graph = nx.erdos_renyi_graph(n=100, p=0.6, directed=True, seed=42)
for u, v in graph.edges():
    try:
        graph[u][v]['weight'] += 1
    except:
        graph.add_edge(u, v, weight=1)
print('Built graph G')
#graph = nx.read_edgelist(direct + "\\obama.txt")
'''
# ======================================================================================================================
# Real dataset (ca-CSphd.txt)
# ======================================================================================================================

# read in graph
graph = nx.DiGraph()
with open(direct + "\\ca-CSphd.txt") as f:
    n, m = f.readline().split()
    for line in f:
        u, v = map(int, line.split())
        try:
            graph[u][v]['weight'] += 1
        except:
            graph.add_edge(u, v, weight=1)
print('Built graph G')


#with open(direct + "\\ca-CSphd.txt") as f:
#with open(direct + "\\obama.txt") as f:
    #lines = f.readlines()

#myList = [line.strip().split() for line in lines]
# [['a', 'b'], ['a', 'c'], ['b', 'd'], ['c', 'e']]

# ======================================================================================================================

#graph = nx.Graph() # UNDIRECTED GRAPH

#graph = nx.DiGraph() # DIRECTED GRAPH
#graph.add_edges_from(myList)


# ======================================================================================================================
# Print Graphs
# ======================================================================================================================

# draw labeled network
nx.draw(graph, with_labels=True, font_weight='bold')
print("test draw network")
# ======================================================================================================================

plt.figure(figsize=(16,10))
color_map = {1: 'b', 0: 'r'}
pos = nx.random_layout(graph)
nx.draw_networkx(graph, pos, with_labels=False, node_size=100, node_shape='.',
                linewidth=None, width=0.2, edge_color='y')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks([])
plt.yticks([])
plt.show()

# ======================================================================================================================

print(graph.nodes())
# ['a', 'c', 'b', 'e', 'd']
print(graph.edges())
# [('a', 'c'), ('a', 'b'), ('c', 'e'), ('b', 'd')]

# ======================================================================================================================
# Number of nodes targeted
targeted_size = 100
# ======================================================================================================================
all_nodes = len(graph.nodes()) / 10
# ======================================================================================================================

# binaryDegreeDiscount with degreeDiscount
start1 = time.time()
S1, sizeSpread1, timer1 = binaryDegreeDiscount(graph, all_nodes, step=50, degreeDiscount_Heuristic='degreeDiscount')
print('Necessary %s initial nodes to target %s nodes in graph G' % (len(S1), targeted_size))
print(S1)
time1 = time.time() - start1
print(time1)

print("binaryDegreeDiscount with degreeDiscount  ESCAPED")

# binaryDegreeDiscount with degreeHeuristic
start2 = time.time()
S2, sizeSpread2, timer2 = binaryDegreeDiscount(graph, all_nodes, step=50, degreeDiscount_Heuristic='degreeHeuristic')
print('Necessary %s initial nodes to target %s nodes in graph G' % (len(S2), targeted_size))
print(S2)
time2 = time.time() - start2
print(time2)

print("binaryDegreeDiscount with degreeHeuristic  ESCAPED")


# spreadDegreeDiscount with degreeDiscount
start3 = time.time()
S3, sizeSpread3, timer3 = spreadDegreeDiscount(graph, all_nodes, step=50, degreeDiscount_Heuristic='degreeDiscount')
print('Necessary %s initial nodes to target %s nodes in graph G' % (len(S3), targeted_size))
print(S3)
time3 = time.time() - start3
print(time3)

print("spreadDegreeDiscount with degreeDiscount  ESCAPED")

# spreadDegreeDiscount with degreeHeuristic
start4 = time.time()
S4, sizeSpread4, timer4 = spreadDegreeDiscount(graph, all_nodes, step=50, degreeDiscount_Heuristic='degreeHeuristic')
print('Necessary %s initial nodes to target %s nodes in graph G' % (len(S4), targeted_size))
print(S4)
time4 = time.time() - start4
print(time4)

print("spreadDegreeDiscount with degreeHeuristic  ESCAPED")
# ======================================================================================================================

#print("spread - 1 seed: ", IndependentCascadeModel(graph, S1, propagation_proba=0.1, monte_carlo_sim=1000))
#print("spread - 2 seed: ", IndependentCascadeModel(graph, S2, propagation_proba=0.1, monte_carlo_sim=1000))
#print("spread - 3 seed: ", IndependentCascadeModel(graph, S3, propagation_proba=0.1, monte_carlo_sim=1000))
#print("spread - 4 seed: ", IndependentCascadeModel(graph, S4, propagation_proba=0.1, monte_carlo_sim=1000))
#print("spread - 4 seed: ", IndependentCascadeModel(graph, [1454, 2708, 2490], propagation_proba=0.1, monte_carlo_sim=1000))
#print("spread - 7 seed: ", IndependentCascadeModel(graph, [1454, 2708, 3016, 1613, 2283, 2490, 3056], propagation_proba=0.1, monte_carlo_sim=1000))

# ======================================================================================================================
# Run algorithms

celf_influential_nodes, celf_spread, celf_time = celf_algorithm(graph, targeted_size, propagation_proba=0.1, monte_carlo_sim=1000)
print("CELF ESCAPED")

greedy_influential_nodes, greedy_spread, greedy_time  = greedy_influence_maximization(graph, targeted_size, propagation_proba=0.1, monte_carlo_sim=1000)
print("GREEDY ESCAPED")

# ======================================================================================================================

# Print resulting seed sets
print("celf output:   " + str(celf_influential_nodes))
print("greedy output: " + str(greedy_influential_nodes))

# ======================================================================================================================
# Print the graph highlighting the graph nodes that were selected by each algorithm
# ======================================================================================================================
plt.figure(figsize=(16,10))
color_map = {1: 'b', 0: 'r'}
pos = nx.random_layout(graph)
# for CELF
nx.draw_networkx(graph, pos, nodelist=celf_influential_nodes, node_color='r', with_labels=False, node_size=100, node_shape='.',
                linewidth=None, width=0.2, edge_color='y')
# for Greedy
nx.draw_networkx(graph, pos, nodelist=greedy_influential_nodes, node_color='green', with_labels=False, node_size=100, node_shape='.',
                linewidth=None, width=0.2, edge_color='y')
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.xticks([])
plt.yticks([])
plt.show()

# ======================================================================================================================
plt.figure(figsize=(16,10))

# Plot settings
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False

# Plot Computation Time
plt.plot(range(1, len(greedy_time)+1), greedy_time, label="Greedy", color="#FBB4AE")
plt.plot(range(1, len(celf_time)+1), celf_time, label="CELF", color="#B3CDE3")

plt.plot(range(1, len(timer1)+1), timer1, label="binaryDegreeDiscount-degreeDiscount", color="black") #range(1, len(S1)+1),
plt.plot(range(1, len(timer2)+1), timer2, label="binaryDegreeDiscount-degreeHeuristic", color="darkgreen")
plt.plot(range(1, len(timer3)+1), timer3, label="spreadDegreeDiscount-degreeDiscount", color="teal")
plt.plot(range(1, len(timer4)+1), timer4, label="spreadDegreeDiscount-degreeHeuristic", color="red")


plt.ylabel('Computation Time (Seconds)')
plt.xlabel('Size of Seed Set')
plt.title('Computation Time')
plt.legend(loc=2)

plt.show()

# ======================================================================================================================

# Plot Expected Spread by Seed Set Size
plt.plot(range(1, len(greedy_spread)+1), greedy_spread, label="Greedy", color="#FBB4AE")
plt.plot(range(1, len(celf_spread)+1), celf_spread, label="CELF", color="#B3CDE3")

plt.plot(range(1, len(sizeSpread1)+1), sizeSpread1, label="binaryDegreeDiscount-degreeDiscount", color="black")
plt.plot(range(1, len(sizeSpread2)+1), sizeSpread2, label="binaryDegreeDiscount-degreeHeuristic", color="darkgreen")
plt.plot(range(1, len(sizeSpread3)+1), sizeSpread3, label="spreadDegreeDiscount-degreeDiscount", color="teal")
plt.plot(range(1, len(sizeSpread4)+1), sizeSpread4, label="spreadDegreeDiscount-degreeHeuristic", color="red")

plt.xlabel('Size of Seed Set')
plt.ylabel('Expected Spread')
plt.title('Expected Spread')
plt.legend(loc=2)

plt.show()