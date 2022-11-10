#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 03:27:58 2022
@author: Optimus
"""

import networkx as nx
import random
import math
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
from collections import Counter


n=50
p=0.8
random.seed(0)
#fileL = open('twitter-larger.tsv','rb')
##fileS = open('twitter_smallT.csv','rb')
#read from file file
##Gs=nx.read_edgelist(fileS, delimiter='\t',create_using=nx.DiGraph(),data=[('weight',int),('Timestamp',str)])
#fianlly close the inputs stream
##fileS.close()

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
print('Number of nodes {} Number of edges {}'.format(NumberOfNodesC,NumberOfEdgesC))
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
if nx.is_connected(GU):
    print('True')
else:
    lp=[]
    subgraphs=nx.connected_components(GU)
    subgraphs=[sbg for sbg in subgraphs if len(sbg)>1]
    for n in range(1,len(subgraphs)):
        for node in subgraphs[n]: 
            lp.append(nx.single_source_shortest_path_length(GU,node).values())
    lps=[]
    for t in range(len(lp)):
        x=[i for i in lp[t]]
        lps.append(x[-1])
    diameter=np.mean(lps)
#	
#	subgraphs=[sbg for sbg in subgraphs if len(sbg)>1]#
#	for g in subgraphs:
#        print(g)
    #diameter=np.sum(nx.average_shortest_path_length(sg) for sg in subgraphs)/len(subgraphs)
#diameter=nx.diameter(G)
degreeS=sorted(G.degree, key=lambda x: x[1], reverse=True)
print(degreeS[0])
print(degreeS[-1])
dg=np.array(degreeS)
dg=dg[:,1].astype(float)
AverageDegree=np.mean(dg)
#print(density,file=open('density.txt','w'))
print("AC {} density {} diameter {} Average Degree {} nodes {} edges {}  Number of Nodes Component {} Number of Edges Component {}".format(AverageClustering,density,diameter,AverageDegree,Nodes,Edges,NumberOfNodesC,NumberOfEdgesC),file=open('info.txt','w'))

def perturbation(graph, p):
    g = graph.copy()
    edges_to_remove = int(len(g.edges()) * p)
    
    removed_edges = []
    for i in range(edges_to_remove):
        random_edge = random.choice(list(g.edges()))
        g.remove_edges_from([random_edge])
        removed_edges.append(random_edge)

    while(edges_to_remove > 0):
        first_node = random.choice(list(g.nodes()))
        second_node = random.choice(list(g.nodes()))
        if(second_node == first_node):
            continue
        if g.has_edge(first_node, second_node) or (first_node, second_node) in removed_edges or (second_node, first_node) in removed_edges:
            continue
        else:
            g.add_edge(first_node, second_node)
            edges_to_remove -= 1
    
    return g

measures = {}
for pert in [0.05]:#, .05, .1, .2, .5, 1]:
    print('  perturbation ({:.0%} of edges)...'.format(pert))
    pert_graph = perturbation(G, pert)
    
G=pert_graph
#finding the degree
GU=G.to_undirected()
#clustring coefficients
AverageClusteringPert = nx.average_clustering(GU)
densityPert=nx.density(G)
if nx.is_connected(GU):
    diameterPert=nx.diameter(G)
else:
    lp=[]
    subgraphs=nx.connected_components(GU)
    subgraphs=[sbg for sbg in subgraphs if len(sbg)>1]
    for n in range(1,len(subgraphs)):
        for node in subgraphs[n]: 
            lp.append(nx.single_source_shortest_path_length(GU,node).values())
    lps=[]
    for t in range(len(lp)):
        x=[i for i in lp[t]]
        lps.append(x[-1])
    diameterPert=np.mean(lps)
degreeS=sorted(G.degree, key=lambda x: x[1], reverse=True)
print(degreeS[0])
print(degreeS[-1])
dg=np.array(degreeS)
dg=dg[:,1].astype(float)
AverageDegreePert=np.mean(dg)
NumberOfNodesPert = G.number_of_nodes()
NumberOfEdgesPert = G.number_of_edges()
print("AC {} density {} diameter {} Average Degree {} nodes {} edges {}  Number of Nodes Component {} Number of Edges Component {}".format(AverageClusteringPert,densityPert,diameterPert,AverageDegreePert,Nodes,Edges,NumberOfNodesPert,NumberOfEdgesPert),file=open('infoPert.txt','w'))

def eq_class(facts: dict):
    eq_class = {}
    for key, degrees in facts.items():
        k = tuple(sorted(degrees))
        
        if k not in eq_class:
            eq_class[k] = [] # Initialize the value field for that empty key
        
        eq_class[k].append(key)

    return eq_class


def deanonymize(facts, query_name):
    eq = eq_class(facts).values()

    f = lambda vals, minv, maxv: [len(v) for v in vals if len(v) >= minv and len(v) <= maxv]

    deanonymized_nodes = {}
    
    deanonymized_nodes['1'] = f(eq, 1, 1)
    deanonymized_nodes['2-4'] = f(eq, 2, 4)
    deanonymized_nodes['5-10'] = f(eq, 5, 10)
    deanonymized_nodes['11-20'] = f(eq, 11, 20)
    deanonymized_nodes['20-inf'] = f(eq, 20, 'inf')

    tot = sum([vv for v in deanonymized_nodes.values() for vv in v])

    data = pd.Series()
    for k,v in deanonymized_nodes.items():
        data['{} deanonymization [{}]'.format(query_name, k)] = sum(v) / tot
    
    return data

def hi(g, i: int):
    neighbors = {n: knbrs(g, n, i-1) for n in g.nodes()}

    res = {}
    for k,v in neighbors.items():
        if i == 0:
            res[k] = [0]
        else:
            res[k] = sorted([g.degree(n) for n in v])

    return res

def deanonymize_h(g, i):
    h = hi(g, i)

    #print('h', i)
    #print(h)

    return deanonymize(h, 'h({})'.format(i))




def knbrs(g, start, k):
    nbrs = set([start])
    for i in range(k):
        nbrs = set((nbr for n in nbrs for nbr in g[n]))
    return nbrs

h = [deanonymize_h(pert_graph, i) for i in range(0, 5)]
###################
#   deanonymize neighborsOne-Two
##################
#def normalize(a):
#    return a/a.sum()
nodes=G.nodes()
fnei=[]
for n in nodes:
    neig=list(G.neighbors(n))
    neigneig=[len(list(G.neighbors(ne))) for ne in neig]
    degree=G.degree(n)
    degreeT=[G.degree(ne) for ne in neig]
    degreeT=np.array(degreeT)
    degreeT=np.sum(degreeT)
    fnei.append([n,degree,len(neig),neig,degreeT,len(neigneig),neigneig])
fnei=np.array(fnei)
c=Counter(fnei[:,1])
counterO=0
counterT=0
counterTh=0
counterF=0
counterFi=0
OneN=[]
for v in c.values():
    if v >= 1 and v <=1:
        counterO += 1
    if v >=2 and v<=4:
        counterT +=1
    if v >=5 and v<=10:
        counterTh +=1
    if v >=11 and v<=20:
        counterF +=1
    if v >=20:
        counterFi +=1
OneN.append([counterO/len(fnei),counterT/len(fnei),counterTh/len(fnei),counterF/len(fnei),counterFi/len(fnei)])
c=Counter(fnei[:,2])
counterO=0
counterT=0
counterTh=0
counterF=0
counterFi=0
OneT=[]
for v in c.values():
    if v >= 1 and v <=1:
        counterO += 1
    if v >=2 and v<=4:
        counterT +=1
    if v >=5 and v<=10:
        counterTh +=1
    if v >=11 and v<=20:
        counterF +=1
    if v >=20:
        counterFi +=1
OneT.append([counterO/len(fnei),counterT/len(fnei),counterTh/len(fnei),counterF/len(fnei),counterFi/len(fnei)])
c=Counter(fnei[:,4])
counterO=0
counterT=0
counterTh=0
counterF=0
counterFi=0
OneTh=[]
for v in c.values():
    if v >= 1 and v <=1:
        counterO += 1
    if v >=2 and v<=4:
        counterT +=1
    if v >=5 and v<=10:
        counterTh +=1
    if v >=11 and v<=20:
        counterF +=1
    if v >=20:
        counterFi +=1
OneTh.append([counterO/len(fnei),counterT/len(fnei),counterTh/len(fnei),counterF/len(fnei),counterFi/len(fnei)])
