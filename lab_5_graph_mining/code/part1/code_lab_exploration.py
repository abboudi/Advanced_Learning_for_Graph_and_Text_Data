"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
G = nx.read_edgelist("../datasets/CA-HepTh.txt",comments='#',delimiter='\t',create_using=nx.Graph())
print("Edges", G.number_of_edges())
##################



############## Task 2

##################
# your code here #
print("Number of connected components", nx.number_connected_components(G))
gcc_nodes = max(nx.connected_components(G),key=len)
gcc = G.subgraph(gcc_nodes)
print("Fraction of nodes in gcc : ", gcc.number_of_nodes()/G.number_of_nodes())
print("Fraction of edges in gcc : ", gcc.number_of_edges()/G.number_of_edges())
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
min_degree = np.min(degree_sequence)
max_degree = np.max(degree_sequence)
median_degree = np.median(degree_sequence)
mean_degree = np.mean(degree_sequence)

print("Min degree : ", min_degree)
print("Max degree : ", max_degree)
print("Median degree : ", median_degree)
print("Mean degree : ", mean_degree)

##################



############## Task 4

##################
# your code here #
hist = nx.degree_histogram(G)
plt.plot(hist)
plt.title("Degree distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.show()
hist = nx.degree_histogram(G)
plt.loglog(hist)
plt.title("Log of degree distribution")
plt.xlabel("log(Degree)")
plt.ylabel("log(Frequency)")
plt.show()
##################
