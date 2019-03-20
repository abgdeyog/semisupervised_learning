import networkx as nx
from networkx.algorithms import node_classification
import numpy as np
import pandas as pd
import math
import random
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

def find_sigma(Graph, label_name):
    sigma = 0.5
    sigma_new = 0.6
    precision = 0
    precision_new = 0
    step_multiplier = 0.01
    while abs(sigma - sigma_new) > 10e-6:
        weights = {}
        for key in Graph.nodes:
            shortest_path = nx.algorithms.shortest_path(Graph, key)
            for dest in shortest_path:
                weight = math.exp(-len(shortest_path[dest]))
                weights[(key, dest)] = weight
        for key_pair in weights:
            weights[key_pair] *= sigma
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
        fl_train = {}
        fl_test = {}
        fu_train = fu.copy()
        for i in fl:
            if random.random() > 0.3:
                fl_train[i] = fl[i]
            else:
                fl_test[i] = fl[i]
                fu_train[i] = fl[i]

        fu_train_list = [key for key in fu] + [key for key in fl_test]
        fl_train_list = [key for key in fl_train]
        fl_test_list = [key for key in fl_test]
        Duu = np.zeros((len(fu_train), len(fu_train)))
        Wuu = np.zeros((len(fu_train), len(fu_train)))
        Wul = np.zeros((len(fu_train), len(fl_train)))
        row = 0
        for u_index in fu_train_list:
            column = 0
            for l_index in fl_train_list:
                try:
                    Wul[row][column] = weights[(u_index, l_index)]
                except KeyError:
                    pass
                column += 1
            column = 0
            for u_index2 in fu_train_list:
                try:
                    Wuu[row][column] = weights[(u_index, u_index2)]
                except KeyError:
                    pass
                column += 1
            row += 1
        for row in range(len(fu_train)):
            sum = 0
            for column in range(len(fu_train)):
                sum += Wuu[row][column]
            for column in range(len(fl_train)):
                sum += Wul[row][column]
            Duu[row][row] = sum
        fl_values = np.array([fl[key] for key in fl_train])
        fu_values = np.dot(np.dot(np.linalg.inv(Duu - Wuu), Wul), fl_values)

        for row in range(len(fu_values)):
            fu_train[fu_train_list[row]] = fu_values[row]
        labeled_list = []
        for i in range(len(fu_train) + len(fl_train)):
            try:
                labeled_list.append(fl_train[i])
            except KeyError:
                labeled_list.append(fu_train[i])
        correct = 0
        for key in fl_test:
            if int(round(fu_train[key])) == fl_test[key]:
                correct += 1
        precision = precision_new
        precision_new = correct / len(fl_test)
        step = sigma_new - sigma
        sigma = sigma_new
        sigma_new = sigma + (precision_new - precision)/step * step_multiplier
        print(sigma)
    return sigma_new

def your_harmonic_function(Graph, label_name):
    # YOUR CODE HERE
    sigma = find_sigma(Graph, label_name)
    print(sigma)
    weights = {}
    for key in Graph.nodes:
        shortest_path = nx.algorithms.shortest_path(Graph, key)
        for dest in shortest_path:
            weight = math.exp(-len(shortest_path[dest]))
            weights[(key, dest)] = weight
    for key_pair in weights:
        weights[key_pair] *= sigma
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