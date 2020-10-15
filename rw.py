# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 18:28:54 2017

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
from collections import defaultdict
import itertools
import time
import sys
import math

print("the name of this file is:"+ sys.argv[0],"Number of Crawled :"+ sys.argv[1])
#run file e-pinions1
file = open('release-youtube-links','rb')
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


# =============================================================================
# Random Walk
# =============================================================================
#find all nodes
nodes = G.nodes()
#find all edges
edges = G.edges()
#sample
samp =[]
#number of seed
k=1
#how many nodes to crawl
threshold = int(sys.argv[1])
ind_pos=sample(range(len(nodes)),k)
Vseed=list(itemgetter(*ind_pos)(nodes))
#i use 1 seed
v = Vseed.pop(0)
#add it sample
samp.append(v)
#initial node
v1 = v
# coutner
c = 0
#number of iterations
it = 0
tt = 0
#how many times i get stuck
dead=0
#time
start = time.time()
#do while i have a good size
while(c <threshold):
    #randomly and uniformly 0-1 probabilit
    pr = round(np.random.uniform(0, 1),2)
    #if probability > 0.15 add the neighbor in the sample
    if pr > 0.15 and len(G.neighbors(v)) !=0:
        #choose a beighbor
         w = choice(G.neighbors(v))
         #if not in the sample add
         if w not in samp:
            samp.append(w)
            #next node is now the parent
            v=w
            #add in counter
            c += 1
    #if probability <= 15 go back to initial node v1
    if pr <= 0.15 and len(G.neighbors(v1)) !=0:
        #choise a neighbor of intial node v1
         f = choice(G.neighbors(v1))
         #if not in the sample add it
         if f not in samp:
             samp.append(f)
             #next node is the parent now
             v=f
             c += 1
    #if all node in sample stuck add iteration
    else :
        it += 1
    #if iterations is more than the sample choose another node
    if it > c:
        v1 = choice(nodes)
        tt = 0
        it = 0
        #how may times jumped
        dead += 1
#end time
end = time.time()
#edges of sample 
ha =G.edges(samp)
#create a digraph
G1 = nx.DiGraph() 
#add edges of sample 
G1.add_edges_from(ha)
#info of the sample
nx.info(G1)
#density
Den = nx.density(G1)
#average degree sample
ADD = sum(G1.degree().values())/float(len(G1))
#average in-out degree
AD = ADD/2
#convert to undirected
Gu = G1.to_undirected()
#average degree unidrected sample
AD1 = sum(Gu.degree().values())/float(len(Gu))
#clustering of the sample
AC=nx.average_clustering(Gu, nodes=None, weight=None, count_zeros=True)
#info of the undirected
nx.info(Gu)
#abstract original-sample
erd =abs(avd_or - ADD)
erc =abs(avc_or - AC) 
#find the percent
percent_thres = threshold*100/len(nodes)
percent_nodes = len(np.unique(ha))*100/len(nodes)
percent_edges = len(ha)*100/len(edges)
sys.stdout = open('rwresults.txt','a')
print('Crawled:10%'
      '\nNumber of seed:',k,
      '\nNumber of crawled:',threshold,"%.2f" % percent_thres,
      '\nNumber of nodes:',len(np.unique(ha)),"%.2f" % percent_nodes,
      '\nNumber of edges:',len(ha),"%.2f" % percent_edges,
      '\nAverage degree (Di) :',AD,
      '\nAverage degree :',ADD,"%.2f" % erd,
      '\nAverage degree(UnD) :',AD1,
      '\nAverage clustering coefficients:',AC,"%.4f" % erc,
      '\nDensity of the graph:',Den,
      '\nNumber of jumps:',dead,
      '\nTime:',(end - start),'\n')
