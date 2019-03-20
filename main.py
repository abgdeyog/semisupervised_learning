import networkx as nx
from networkx.algorithms import node_classification
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def create_toroidal(k):
    G=nx.Graph()
    count = 0
    mapping1 = {}
    mapping2 = {}
    for i in range(k):
        for j in range(k):
            mapping1[count] = (i,j)
            mapping2[(i,j)] = count
            G.add_node(count)
            count +=1
    for i in range(k):
        for j in range(k):
            if (i,j+1) in mapping2:
                G.add_edge(mapping2[(i,j)],mapping2[i,j+1])
            if (i,j-1) in mapping2:
                G.add_edge(mapping2[(i,j)],mapping2[i,j-1])
            if (i+1,j) in mapping2:
                G.add_edge(mapping2[(i,j)],mapping2[i+1,j])
            if (i-1,j) in mapping2:
                G.add_edge(mapping2[(i,j)],mapping2[i-1,j])
    j = 0
    for i in range(k):
        G.add_edge(mapping2[(i,j)],mapping2[i,j+k-1])
    i = 0
    for j in range(k):
        G.add_edge(mapping2[(i,j)],mapping2[i+k-1,j])
    return G

def your_harmonic_function(Graph, label_name):
    # YOUR CODE HERE
    x = nx.spring_layout(G, iterations=200)
    weights = {}
    for edge in Graph.edges:
        weights[edge] = 1
    fu = {}
    fl = {}
    uniq_labeles = {}
    uniq_labeles_n = 0
    uniq_labeles_inverse = []
    for key in Graph.nodes:
        try:
            label = Graph.node[key][label_name]
            if label not in uniq_labeles:
                uniq_labeles[label] = uniq_labeles_n
                uniq_labeles_inverse.append(label)
                uniq_labeles_n += 1
            fl[key] = uniq_labeles[label]
        except KeyError:
            fu[key] = ""
    fu_list = [key for key in fu]
    fl_list = [key for key in fl]
    Duu = np.zeros((len(fu), len(fu)))
    Wuu = np.zeros((len(fu), len(fu)))
    Wul = np.zeros((len(fu), len(fl)))
    row = 0
    for u_index in fu_list:
        column = 0
        for l_index in fl_list:
            try:
                Wul[row][column] = weights[(u_index, l_index)]
            except KeyError:
                pass
            column += 1
        column = 0
        for u_index2 in fu_list:
            try:
                Wuu[row][column] = weights[(u_index, u_index2)]
            except KeyError:
                pass
            column += 1
        row += 1
    for row in range(len(fu)):
        sum = 0
        for column in range(len(fu)):
            sum += Wuu[row][column]
        for column in range(len(fl)):
            sum += Wul[row][column]
        Duu[row][row] = sum
    fl_values = np.array([fl[key] for key in fl_list])
    fu_values = np.dot(np.dot(np.linalg.inv(Duu - Wuu), Wul), fl_values)
    for row in range(len(fu_values)):
        fu[fu_list[row]] = fu_values[row]
    labeled_list = []
    for i in range(len(fu) + len(fl)):
        try:
            labeled_list.append(fl[i])
        except KeyError:
            labeled_list.append(fu[i])

    return [uniq_labeles_inverse[int(round(value))] for value in labeled_list]


G = create_toroidal(16)
G.node[0]['label'] = 'blue'
G.node[255]['label'] = 'red'
gpos = nx.spring_layout(G, iterations=200)
node_color=['blue' if n == 0 else 'red' if n == 255 else 'gray' for n in G.nodes]
plt.figure(figsize=(10,10))
nx.draw(G, gpos, with_labels=False, node_size=200, node_color=node_color)
plt.show()

node_color=node_classification.harmonic_function(G)
removed = [n for n in G.nodes if n%3 == 0 ]
for n in G.nodes:
    if n not in removed:
        G.node[n]['label'] = node_color[n]
predicted = your_harmonic_function(G, label_name='label')

print(confusion_matrix(node_color, predicted))
print(precision_recall_fscore_support(node_color, predicted))