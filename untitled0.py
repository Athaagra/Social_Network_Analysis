#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:27:02 2022

@author: Optimus
"""

import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.community import greedy_modularity_communities



import networkx as nx
from networkx.algorithms import community

import networkx as nx
import pandas as pd

fileS = pd.read_csv('giant_componentnSmall2022.csv',sep='\t')

#fileS = open('giant_componentnSmall2022.csv','rb')
#read from file file

#fileS.close()
titles=list(fileS.columns)
G = nx.from_pandas_edgelist(fileS,titles[0],titles[1],[titles[2],titles[3]])



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