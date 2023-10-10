#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:19:42 2023

@author: Optimus
"""

#Implementation of some common vector operations
#that are needed in both algorithms
import networkx as nx
import numpy as np

def unit_vector(v):
    #unit vector computation
    v = v.getA1()#handling type mismatch
    temp = [x*x for x in v]
    #print(temp)
    denom = sum(temp) ** 0.5
    unit_vec = []
    for i in range(len(v)):
        unit_vec.append(round(v[i]/denom, 5))
    unit_vec = np.matrix(unit_vec).getT()
    return unit_vec

def normalized(a):
    #returns magnitude
    a = a.getA1()# type compatibility
    return round(sum([x*x for x in a])**0.5, 5)

def leading_eigen_vector(adjacency_matrix):
    #leading eigen vector computation using power method
    order = adjacency_matrix.shape[0]
    #initial vector with all +ve components
    eigen_vector = [3, 2] * (order // 2)
    eigen_vector += [1] * (order % 2)
    eigen_vector = np.matrix(eigen_vector).getT()#34x1 matrix
    result = np.dot(adjacency_matrix, eigen_vector)
    normalized_val = normalized(result)
    while(True):
        eigen_vector = unit_vector(result)
        #print(eigen_vector)#ok
        result = np.dot(adjacency_matrix, eigen_vector)
        current_normalized_val = normalized(result)
        if normalized_val - current_normalized_val == 0:
            break
        normalized_val = current_normalized_val
        #print(normalized_val)
    return eigen_vector

def eigenVec_2nd_smallest_eigenVal(lev, l_mat):
    #calculates eigen vector corresponding to 2nd smallest eigen value of a matrix
    order = lev.shape[0]
    #initial vector with all +ve components
    x = [1, 2] * (order//2)
    x += [3] * (order % 2)
    x = np.matrix(x).getT()#creating 34x1 vector
    c = float(np.dot(lev.getT(), x))
    y = x - c * lev
    it = 500
    while(it > 0):
        y = np.dot(l_mat.getT(), y)
        if it % 10 == 0:
        #removing component of largest ev from y periodically       
            c = float(np.dot(lev.getT(), y))
            y = y - c * lev
        it -= 1
        y = unit_vector(y)
    return y

def get_partition(G, fv):
    #partitioning based on minimum cut size
    edges = G.edges()
    #sorting the vertices based on components of fiedler vector
    c = sorted(range(len(fv)), key=lambda k: fv[k])
    #partition 1
    c11 = set(i+1 for i in c[:c[1]])#pick 1st 16
    c12 = set(j+1 for j in c[c[1]:])# pick rest i.e 18
    #partition 2
    c21 = set(j+1 for j in c[-c[1]:])#pick last 16
    c22 = set(j+1 for j in c[:-c[1]])#pick rest
    # minimize cut size
    cut_size_1 = 0
    cut_size_2 = 0
    for e in edges:
        u = int(e[0]) 
        v = int(e[1])
        if (u in c11 and v in c12) or (u in c12 and v in c11):
            cut_size_1 += 1
        if (u in c21 and v in c22) or (u in c22 and v in c21):
            cut_size_2 += 1
    #return partition with min cut size
    if cut_size_1 < cut_size_2:
        return c11, c12
    else: 
        return c21, c22
    
    
#n = 1000  # 1000 nodes
#m = 5000  # 5000 edges
#G = nx.gnm_random_graph(n, m)
#G = nx.karate_club_graph()
G=nx.read_edgelist('cit-hep.txt', delimiter='\t',create_using=nx.DiGraph(),data=[('weight',int),('Timestamp',str)])
G = G.to_undirected(G)
m = nx.linalg.modularitymatrix.modularity_matrix(G)
no_of_vertices = m.shape[0]
lev = leading_eigen_vector(m).getA1()
c1 = set()
c2 = set()
# assigning community based on sign
for i in range(no_of_vertices):
		if lev[i] < 0:
			c2.add(i+1)
		else:
			c1.add(i+1)
print("Community discovery by Modularity Maximization ---- ")
print("Community 1 :", c1)
print("Community 2 :", c2)
print("Size : ",len(c1), len(c2), "respectively")

def a_matrix(G, alpha):
    n_nodes = len(G.nodes())
    a_nodes=np.array(G.nodes())
    edges = G.edges()
    # building adjacent matrix
    adj_matrix = np.zeros(shape=(n_nodes, n_nodes))
    #for edge in G.edges():
    #    adj_matrix[edge[0], edge[1]] = 1
    for i in range(n_nodes):
        index_n=np.where(a_nodes==a_nodes[i])
        n_node_edges=np.array(list(G.neighbors(a_nodes[i])))
        for j in range(len(n_node_edges)):
            index_nn=np.where(a_nodes==a_nodes[j])
            adj_matrix[index_n,index_nn]=1
    edges_l=list(edges)
    edges_l.append([1000,1000])
    c=0
    edge_su=[]
    for i in range(0,len(edges_l)):
        if edges_l[i][0]==edges_l[i-1][0]:
            c+=1
        else:
            edge_su.append([edges_l[i-1][0],c])
            c=1
    edge_su=edge_su[1:]
    
    degree_matrix = np.zeros(shape=(n_nodes, n_nodes))
    for i in edge_su:
        degree_matrix[np.where(a_nodes==i[0]),np.where(a_nodes==i[0])]=i[1]
    L=adj_matrix+degree_matrix
    return adj_matrix,degree_matrix,L

def edge_conductance(c1,c2):
    c1=list(c1)
    c2=list(c2)
    a_nodes=np.array(G.nodes())
    edges = G.edges()
    ed=list(edges)
    shared_con=0
    c2_ind=np.arange(0,len(c2))
    c2_ind=np.tile(c2_ind,len(c1))
    c2_ind=c2_ind.reshape(len(c2_ind),1)
    c1_ind=np.arange(0,len(c1))
    c1_ind=np.repeat(c1_ind,len(c2))
    c1_ind=c1_ind.reshape(len(c1_ind),1)
    c1c2_ind=np.hstack((c1_ind,c2_ind))
    for q in range(len(c1c2_ind)):
        #print(c1c2_ind[q])
        if (a_nodes[c1c2_ind[q][0]],a_nodes[c1c2_ind[q][1]]) in ed or (a_nodes[c1c2_ind[q][1]],a_nodes[c1c2_ind[q][0]]) in ed:
            shared_con+=1
    edc1=0
    edc2=0
    for e in edges:
        if e[0] in c1:
            edc1+=1
        else:
            edc2+=1
    resedgecond=shared_con/min(edc1,edc2)
    return resedgecond

pg = a_matrix(G, 1)
#L = nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()
adjac,degree,Lap=a_matrix(G,0)
n = G.number_of_nodes()
k_max = max(G.degree())[1]#max degree 
I = np.matrix(np.identity(n))
m = (2 * k_max * I) - Lap
lev = leading_eigen_vector(m)
fv = eigenVec_2nd_smallest_eigenVal(lev, m)
#print(fv)
c1, c2 = get_partition(G, fv)
conductance=edge_conductance(c1, c2)
print("Spectral partitioning using Fiedler vector -----")
print("\n......Own implementation.......")
print("Community 1 : ", c1)
print("Community 2 : ", c2)
print("Size : ",len(c1), " and ", len(c2), "respectively")
print("Conductance {}".format(conductance))
#same thing using library function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# reading the data and looking at the first five rows of the data
data=pd.read_csv("Wholesale customers data.csv")
data.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)
#kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
#kmeans.fit(data_scaled)
#pred = kmeans.predict(data_scaled)


kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()
