# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:14:22 2017

@author: Kel3vra

"""
from collections import Counter
import operator
import csv
import functools
import collections
import queue as que
from operator import itemgetter
from numpy.random import randint
from random import choice
from random import sample
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import itertools
import time
import sys
import math
from sklearn.metrics import mean_squared_error

#run file e-pinions1
file = open('soc-Epinions1.txt','rb')
#create a directed graph
G = nx.read_edgelist(file,create_using=nx.DiGraph())
#info of the graph
nx.info(G)
#average degree of Epinions
avd_or = sum(G.degree().values())/float(len(G))
#convert to undirected
Ga = G.to_undirected()
#average degree of undirected
AD1 = sum(Ga.degree().values())/float(len(Ga))
#average cluster_coefficients of epinions
avc_or=nx.average_clustering(Ga, nodes=None, weight=None, count_zeros=True)
#info of undirected epinions
nx.info(Ga)






#nodes i have crawled FIFO
Qcrawl = []
#nodes i seen --neighbors
Vseen=[]
#node i have crawled sampled
Vcrawl = []
#all the nodes
nodes = G.nodes()
#all the edges
edges = G.edges()
#number of seeds
k=100
#how many i want to sample
threshold = 61176
#random and uniformly i select the seeds
ind_pos=sample(range(len(nodes)),k)
Vseed=list(itemgetter(*ind_pos)(nodes))
for i in Vseed:
    Qcrawl.append(i)
    Vseen.append(i)
#iterations
c = 0
#time i calculate
start = time.time()
#how many iterations
while(c<threshold):
    #pop the First of the list
    u = Qcrawl.pop(0)
    #if not in the sampel i add and count 1
    if u not in Vcrawl:
        Vcrawl.append(u)
        c += 1
    #neighbors
    Ou = G.neighbors(u)
    #i add the neighbor in the FIFO and Vseen
    for x in Ou:
        if x not in Vseen :
            Qcrawl.append(x)
            Vseen.append(x)
#end time
end = time.time() 
#my new graph from the sample          
f =G.edges(Vseen)
#i created as directed
G1 = nx.DiGraph()
#i add the edges in the graph sample 
G1.add_edges_from(f)
#density of the graph sample
Den = nx.density(G1)
#average degree of direceted sample
ADD = sum(G1.degree().values())/float(len(G1))
#average degree in-out degree sample
AD = ADD/2
#convert to undirected
Gu = G1.to_undirected()
#degree of undirected
AD1 = sum(Gu.degree().values())/float(len(Gu))
#average clestering coefficients sample
AC=nx.average_clustering(Gu, nodes=None, weight=None, count_zeros=True)
#info sample undirected
nx.info(Gu)
#abstract ground truth from sample
erd =abs(avd_or - ADD)
erc =abs(avc_or - AC) 
#find how percent did i find of nodes-edges-crawled
percent_thres = threshold*100/len(nodes)
percent_nodes = len(np.unique(f))*100/len(nodes)
percent_edges = len(f)*100/len(edges)
#add all the results in txt result
sys.stdout = open('results.txt','a')
print('Crawled:10%',
      '\nNumber of seed:',k,
      '\nNumber of crawled:',threshold,"%.2f" % percent_thres,
      '\nNumber of nodes:',len(np.unique(f)),"%.2f" % percent_nodes,
      '\nNumber of edges:',len(f),"%.2f" % percent_edges,
      '\nAverage degree (Out-in) :',AD,
      '\nAverage degree  :',ADD,"%.2f" % erd,
      '\nAverage degree(Und) :',AD1,
      '\nAverage clustering coefficients:',AC,"%.4f" % erc,
      '\nDensity of the graph:',Den,
      '\nTime:',(end - start),'\n')
