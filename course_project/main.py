#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:27:02 2022

@author: Optimus
"""

import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import operator
import networkx as nx
import pandas as pd
import numpy as np
import random

n=5000
p=0.8
random.seed(0)

G=nx.erdos_renyi_graph(n, p, seed=0, directed=True)
NumberOfNodes = G.number_of_nodes()
NumberOfEdges = G.number_of_edges()
print('Number of nodes {} Number of edges {}'.format(NumberOfNodes,NumberOfEdges))
#calculate the strongly components
Strong = nx.number_strongly_connected_components(G)
#calculate the weakly components
Weak = nx.number_weakly_connected_components(G)
print('Number of Strong components {} Number of Weakly Components {}'.format(Strong,Weak))

#calculate the subgraph of strong components
StrongComponent = list(nx.strongly_connected_components(G))
LargestStrong = max(StrongComponent, key=len)
WeaklyComponent = list(nx.weakly_connected_components(G))
largestWeakly = max(WeaklyComponent, key=len)

#finding the degree
GU=G.to_undirected()

#clustring coefficients
AverageClustering = nx.average_clustering(GU)
print("This is AC:{}".format(AverageClustering))
print("This is info:{}".format(nx.info(G)))
print("This is Density:{}".format(nx.density(G)))
print("This is the diameter {}".format(nx.diameter(G)))
degreeS=sorted(G.degree, key=lambda x: x[1], reverse=True)
print(degreeS[0])
print(degreeS[-1])
dg=np.array(degreeS)
AverageDegree=np.mean(dg[:,1])
print(AverageDegree)
