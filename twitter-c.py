# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:07:35 2017

@author: Kel3vra
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import csv
import operator



G=nx.read_edgelist('output.txt', delimiter='\t',create_using=nx.DiGraph(),data=[('Timestamp',str)])
#G=nx.read_edgelist('twitter_small.csv', delimiter='\t',create_using=nx.DiGraph(),data=[('weight',int),('Timestamp',str)])

#calculate the number of nodes
G.number_of_nodes()
#calulate the number of edges
G.number_of_edges()
nx.info(G)
nx.density(G)

#finding the degree
degree_sequence=sorted(nx.degree(G).values(),reverse=True)
dmax=max(degree_sequence)
#plot  a figure a size 10,8 with blue color and the symbol ^
plt.figure(figsize=(10,6))
plt.loglog(degree_sequence,linestyle='--',color='b',marker='^')
#label of diagram
plt.title("degree plot")
#label of y
plt.ylabel("frequence")
#label of x
plt.xlabel("degree")
#save the figure
plt.savefig("degree_histogram.png")
#display the plot
plt.show()

#finding the indegree and outdegree
indegree_sequence=sorted(G.in_degree().values(),reverse=True) 
odegree_sequence=sorted(G.out_degree().values(),reverse=True) 
odmax=max(odegree_sequence)
dmax=max(indegree_sequence)
#figure size
plt.figure(figsize=(10,6))
#both distribution in one diagram
plt.loglog(indegree_sequence,linestyle='--',color='g',marker='^')
plt.loglog(odegree_sequence,linestyle='--',color='r',marker='<')
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

#find the weakly connected component
F=list(nx.weakly_connected_component_subgraphs(G))
largest = max(F,key=len)
nx.info(largest)
#list
frequency=[]
frq={}
#looking for the paths of the largest path
for i,w in enumerate (largest):
    
    DD=sorted(nx.single_source_shortest_path_length(G,w).values())
    
    frq[i]=DD
a={}
for k,v in frq.items():
    if v not in a:
        a[v]=[]
    if v == a[v]:
        c=0
        c = c +1
        a[v].append([c])
#a unit the list frequency
a=np.hstack(sorted(frequency))
#plot the distance distribution
plt.hist(a,bins=60,color='red')
plt.title("Distance distribution")
plt.ylabel("frequency")
plt.xlabel("distance")
plt.savefig("distance_distirbution.png")
plt.show()


     
lar=[]
ve=[]
#find the largest components and edges
for u,v in largest.edges():
      largest1=str(u)
      largest2=str(v)
      ve1 =largest.get_edge_data(u,v)
      ve1 =list(ve1.values())
      lar.append([largest1,largest2,ve1])
lar2=[]
#clear the data
for x in lar:
    user = x[0]
    ment = x[1]
    wei = x[2][0]
    time = x[2][1]
    lar2.append([user,ment,wei,time])
    
    
#create a csv with giant component
f=open('giant_component.csv','w')
f.write('Source \t Target \t Weight \t Timestamp\n') 
writer = csv.writer(f,delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
for line in lar2:
    
    writer.writerow(line)
f.close()
        
#find betweeness centrality
bet = nx.betweenness_centrality(G)
#find closeness centrality
clo =nx.closeness_centrality(G, u=None, distance=None, normalized=True)
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

datal=[]
datan=[]
#clear the data in order to plot
for x in sorted_bet:
    label=x[0]
    data= x[1]
    datal.append(label)
    datan.append(data)
    
    
#plot the betwness centrality
centers = range(len(sorted_bet))
plt.bar(centers, datan, align='center',tick_label=datal)
plt.xticks(rotation=90)
plt.autofmt_xdate()
plt.show()


datal=[]
datan=[]
#clear the data of closeness centrality
for x in sorted_clo:
    label=x[0]
    data= x[1]
    datal.append(label)
    datan.append(data)
#plot the closeness centrality
centers = range(len(sorted_clo))
plt.bar(centers, datan, align='center',tick_label=datal)
plt.xticks(rotation=90)
plt.autofmt_xdate()
plt.show()

datal=[]
datan=[]
#clear the indegree data
for x in sorted_indegree:
    label=x[0]
    data= x[1]
    datal.append(label)
    datan.append(data)
#plot indegree centrality
centers = range(len(sorted_indegree))
plt.bar(centers, datan, align='center',tick_label=datal)
plt.xticks(rotation=90)
plt.autofmt_xdate()
plt.show()
    
    