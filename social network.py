# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:00:02 2017

@author: Kel3vra
"""
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
# open the file medium.in and large.in to read
file = open('soc-Epinions1.txt','rb')
#read from it
G1 = nx.read_edgelist(file,create_using=nx.DiGraph())
#finally close the input stream
file.close()


#calculate the number of nodes
G.number_of_nodes()
#calulate the number of edges
G.number_of_edges()

#calculate the strongly components
nx.number_strongly_connected_components(G)
#calculate the weakly components
nx.number_weakly_connected_components(G)

#calculate the subgraph of strongly components
H= list(nx.strongly_connected_component_subgraphs(G))
#
strong = max(H,key=len)
#display nodes the strongest components 
strong.nodes()
#display edges the strongest components
strong.edges()
#display all the nodes/edges/degree average and indegree average of strongly components
nx.info(strong)

#finding the degree
degree_sequence=sorted(nx.degree(G).values(),reverse=True)
dmax=max(degree_sequence)
#plot  a figure a size 10,8 with blue color and the symbol ^
plt.figure(figsize=(10,8))
plt.semilogy(degree_sequence,'b',marker='^')
#label of diagram
plt.title("Indegree plot")
#label of y
plt.ylabel("frequence")
#label of x
plt.xlabel("indegree")
#save the figure
plt.savefig("degree_histogram.png")
#display the plot
plt.show()

#finding the sequence of outdegree
odegree_sequence=sorted(G.out_degree().values(),reverse=True) 
odmax=max(odegree_sequence)
#plot a figure a size 10.8 with red color and symbol o
plt.figure(figsize=(10,8))
plt.semilogy(odegree_sequence,'r',marker='o')
#label of diagram
plt.title("Out Degree plot")
#label of y
plt.ylabel("frequence")
#label of x
plt.xlabel("outdegrees")
#save the figure
plt.savefig("outdegree_histogram.png")
#display the figure
plt.show()


#finding the indegree and outdegree
degree_sequence=sorted(G.in_degree().values(),reverse=True) 
odegree_sequence=sorted(G.out_degree().values(),reverse=True) 
odmax=max(odegree_sequence)
dmax=max(degree_sequence)
#figure size
plt.figure(figsize=(10,8))
#both distribution in one diagram
plt.semilogy(degree_sequence,'b',marker='^')
plt.semilogy(odegree_sequence,'r',marker='<')
#title of diagram
plt.title("Degree plot")
#label y title
plt.ylabel("nodes")
#label x title
plt.xlabel("degree")
#save the figure
plt.savefig("All_degree_histogram.png")
#display the figure
plt.show()


#the largets of weakly components
F=list(nx.weakly_connected_component_subgraphs(G))
largest = max(F,key=len)
#display the nodes of the weakly
largest.nodes()
#display the edges of the largest
largest.edges()

#list
frequency=[]
#looking for the paths of the largest path
for w in largest:
    DD=sorted(nx.single_source_shortest_path_length(G,w).values())
    frequency.append(DD)
#a unit the list frequency
    
a=np.hstack(sorted(frequency))
#plot the 
plt.hist(a,bins=90,color='red')
plt.title("Distance distribution")
plt.ylabel("frequency")
plt.xlabel("distance")
plt.savefig("distance_distirbution.png")
plt.show()


#second way for largest

for w in largest:
    DD=sorted(nx.shortest_path_length(G,w).values())
    for i in :
    print(i)
    #frequency.append(i)

    
    




