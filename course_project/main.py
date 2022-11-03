import networkx as nx
import random

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
#calculate the subgraph of strong components
StrongComponent = list(nx.strongly_connected_components(G))
LargestStrong = max(StrongComponent, key=len)
WeaklyComponent = list(nx.weakly_connected_components(G))
largestWeakly = max(WeaklyComponent, key=len)

lar = []
ve = []
# find the largest components and edges
for u in largestWeakly:
    #largest1 = str(u)
    #attr = G[largest1]
    #neigh=list(attr.keys())
    #v =list(attr.values())
    #print(v)
    #print(neigh)
    #weight = list(v.values())
    #time = list(v[0].values())
    #v = np.array(v[0].values())
    neigh = list(G.neighbors(u))
    #print("This is the neighbors {}".format(v))
    j=0
    while j != len(neigh):
        #newv=v[j]
        neighv = neigh[j]
        #weight = list(v[j].values())[0]
        #time = list(v[j].values())[1]
        #print("This is the parent node {}".format(largest1))
        #print("This is the attr of nodes {}".format(weight))
        #print("This is the neigh node {}".format(neigh))
        #print("This is the time of nodes {}".format(time))
        #ve1 = G.get_edge_data(largest, str(newv))
        #ve1 = list(ve1.values())
        lar.append([u,neighv])
        j=j+1
Nodes=G.number_of_nodes()
Edges=G.number_of_edges()
import csv
# create a csv with giant component
f = open('LargestComponent2022ErdosRenyi.csv', 'w', encoding='utf-8')
f.write('Source \t Target\n')
writer = csv.writer(f, delimiter='\t',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
for line in lar:
    print(line)
    writer.writerow(line)
f.close()

fileS = open('LargestComponent2022ErdosRenyi.csv','rb')
#read from file file
G=nx.read_edgelist(fileS, delimiter='\t',create_using=nx.DiGraph())
#fianlly close the inputs stream
fileS.close()


NumberOfNodesC = G.number_of_nodes()
NumberOfEdgesC = G.number_of_edges()
print('Number of nodes {} Number of edges {}'.format(NumberOfNodes,NumberOfEdges))
#calculate the strongly components
Strong = nx.number_strongly_connected_components(G)
#calculate the weakly components
Weak = nx.number_weakly_connected_components(G)
print('Number of Strong components {} Number of Weakly Components {}'.format(Strong,Weak))



#finding the degree
GU=G.to_undirected()

#clustring coefficients
AverageClustering = nx.average_clustering(GU)
#print("This is AC:{}".format(AverageClustering))
#print("This is info:{}".format(nx.info(G)))
density=nx.density(G)
diameter=nx.diameter(GU)
degreeS=sorted(G.degree, key=lambda x: x[1], reverse=True)
print(degreeS[0])
print(degreeS[-1])
dg=np.array(degreeS)
AverageDegree=np.mean(dg[:,1])
print(density,file=open('density.txt','w'))
print("density {} diameter {} Average Degree {} nodes {} edges {}  Number of Nodes Component {} Number of Edges Component {}".format(density,diameter,AverageDegree,Nodes,Edges,NumberOfNodesC,NumberOfEdgesC),file=open('info.txt','w'))

