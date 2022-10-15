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

# Import required libraries
from scipy.stats import kendalltau

fileS = pd.read_csv('giant_componentnSmall2022.csv',sep='\t')

#fileS = open('giant_componentnSmall2022.csv','rb')
#read from file file

#fileS.close()
titles=list(fileS.columns)
G = nx.from_pandas_edgelist(fileS,titles[0],titles[1],[titles[2],titles[3]])


#find betweeness centrality
bet = nx.betweenness_centrality(G)
#find closeness centrality
clo =nx.closeness_centrality(G, u=None, distance=None)
#find the in degree centrality
inndegre = nx.in_degree_centrality(G)
#find the out degree centrality
outdegre = nx.out_degree_centrality(G)
#find the top 20 user betwness centrality
sorted_bet = sorted(bet.items(), key=operator.itemgetter(1),reverse=True)[:20]
#find the top 20 user closeness
sorted_clo = sorted(clo.items(), key=operator.itemgetter(1),reverse=True) [:20]
#find the top 20 user according to indegree
sorted_indegree = sorted(inndegre.items(), key=operator.itemgetter(1),reverse=True)  [:20]
#find the top 20 user according to indegree
sorted_outdegree = sorted(outdegre.items(), key=operator.itemgetter(1),reverse=True)  [:20]

datal=[]
databc=[]
#clear the data in order to plot
for x in sorted_bet:
    label=x[0]
    data= x[1]
    datal.append(label)
    databc.append(data)
plt.figure(figsize=(10,6))    
#plot the betwness centrality
centers = range(len(sorted_bet))
plt.bar(centers, databc, align='center',tick_label=datal)
plt.xticks(rotation=90)
plt.autofmt_xdate()
plt.savefig("betweenness-CentralitySmall.png")
plt.show()


datal=[]
datacc=[]
#clear the data of closeness centrality
for x in sorted_clo:
    label=x[0]
    data= x[1]
    datal.append(label)
    datacc.append(data)
#plot the closeness centrality
plt.figure(figsize=(10,6))
centers = range(len(sorted_clo))
plt.bar(centers, datacc, align='center',tick_label=datal)
plt.xticks(rotation=180)
plt.gcf().autofmt_xdate()
plt.savefig("closenessCentralitySmall.png")
plt.show()

datal=[]
dataic=[]
#clear the indegree data
for x in sorted_indegree:
    label=x[0]
    data= x[1]
    datal.append(label)
    dataic.append(data)
#plot indegree centrality
plt.figure(figsize=(10,6))
centers = range(len(sorted_indegree))
plt.bar(centers, dataic, align='center',tick_label=datal)
plt.xticks(rotation=90)
plt.autofmt_xdate()
plt.savefig("indegreeCentralitySmall.png")
plt.show()

datal=[]
dataoc=[]
#clear the indegree data
for x in sorted_outdegree:
    label=x[0]
    data= x[1]
    datal.append(label)
    dataoc.append(data)
#plot indegree centrality
plt.figure(figsize=(10,6))
centers = range(len(sorted_outdegree))
plt.bar(centers, dataoc, align='center',tick_label=datal)
plt.xticks(rotation=90)
plt.autofmt_xdate()
plt.savefig("outdegreeCentralitySmall.png")
plt.show()


dataccA=np.array(datacc)
databcA=np.array(databc)
dataicA=np.array(dataic)
dataocA=np.array(dataoc)  
# Taking values from the above example in Lists
#X = [1, 2, 3, 4, 5, 6, 7]
#Y = [1, 3, 6, 2, 7, 4, 5]

# Calculating Kendall Rank correlation
corrccbc, _ = kendalltau(dataccA, databcA)
corrccic, _ = kendalltau(dataccA, dataicA)
corrccoc, _ = kendalltau(dataccA, dataocA)
corrbcic, _ = kendalltau(databcA, dataicA)
corrbcoc, _ = kendalltau(databcA, dataocA)
corricoc, _ = kendalltau(dataicA, dataocA)

print('Kendall Rank correlation: bec-clc {} clc-inc{} clc-ouc {} bec-inc {} bec-ouc {} inc-outc {}' .format(corrccbc,corrccic,corrccoc,corrbcic,corrbcoc,corricoc), file=open('CorrelationCentralitites.txt','w'))




NumberOfNodes = G.number_of_nodes()
NumberOfEdges = G.number_of_edges()
print('Number of nodes {} Number of edges {}'.format(NumberOfNodes,NumberOfEdges))
G=G.to_undirected()
def edge_to_remove(graph):
  G_dict = nx.edge_betweenness_centrality(graph)
  edge = ()

  # extract the edge with highest edge betweenness centrality score
  for key, value in sorted(G_dict.items(), key=lambda item: item[1], reverse = True):
      edge = key
      break

  return edge

def girvan_newman(graph):
	# find number of connected components
	sg = nx.connected_components(graph)
	sg_count = nx.number_connected_components(graph)

	while(sg_count == 1):
		graph.remove_edge(edge_to_remove(graph)[0], edge_to_remove(graph)[1])
		sg = nx.connected_components(graph)
		sg_count = nx.number_connected_components(graph)

	return sg


# find communities in the graph
c = girvan_newman(G)

# find the nodes forming the communities
node_groups = []

for i in c:
  node_groups.append(list(i))

print(len(node_groups), file=open('NumberOfCommunities.txt','w'))