"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)



############## Task 5

##################
# your code here #
nx.draw_networkx(G, node_color=y)
plt.show()
##################



############## Task 6
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]



############## Task 7

##################
# your code here #
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
embedding_y_pred = classifier.predict(X_test)
embedding_acc_score = accuracy_score(y_true = y_test, y_pred=embedding_y_pred)
print("Embedding Classifier Accuracy Score :", embedding_acc_score)
##################




############## Task 8

##################
# your code here #
se = SpectralEmbedding(n_components=2, affinity='precomputed')
adjacency_matrix = nx.to_numpy_matrix(G)
spectral_embedding = se.fit_transform(adjacency_matrix)

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = spectral_embedding[idx_train,:]
X_test = spectral_embedding[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

se_y_pred = classifier.predict(X_test)
se_acc_score = accuracy_score(y_true = y_test, y_pred=se_y_pred)
print("Spectral Embedding Classifier Accuracy Score :", se_acc_score)

##################
