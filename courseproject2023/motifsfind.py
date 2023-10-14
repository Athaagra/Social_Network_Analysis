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
#m = nx.linalg.modularitymatrix.modularity_matrix(G)
#no_of_vertices = m.shape[0]
#lev = leading_eigen_vector(m).getA1()
#c1 = set()
#c2 = set()
# assigning community based on sign
#for i in range(no_of_vertices):
#		if lev[i] < 0:
#			c2.add(i+1)
#		else:
#			c1.add(i+1)
#print("Community discovery by Modularity Maximization ---- ")
#print("Community 1 :", c1)
#print("Community 2 :", c2)
#print("Size : ",len(c1), len(c2), "respectively")

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

def edge_conductance(c1,c2,G):
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
    for q in c1c2_ind:
        if (a_nodes[q[0]],a_nodes[q[1]]) in ed or (a_nodes[q[1]],a_nodes[q[0]]) in ed:
            shared_con+=1
            print('edge detected')
    edc1=0
    edc2=0
    for n in range (len(c1)-1):
        v = list(G.neighbors(a_nodes[c1[n]]))
        for ng in v:
            if (a_nodes[c1[n]],ng) in ed: 
                edc1+=1
            else:
                edc1+=0
    for n in range (len(c2)-1):
        v = list(G.neighbors(a_nodes[c2[n]]))
        for ng in v:
            if (a_nodes[c2[n]],ng) in ed: 
                edc2+=1
            else:
                edc2+=0
    resedgecond=shared_con/min(edc1,edc2)
    return resedgecond
def mt_m(G,adjac):
    matrixm=[]
    for nod in range (2,len(G.nodes)):
        for node in range (2,len(G.nodes)-1):
            Tv=np.vstack((adjac[nod-2][node-2:node+1],adjac[nod-1][node-2:node+1]))
            Tre=np.vstack((Tv,adjac[nod][node-2:node+1]))
            mtr=sum(sum(Tre))
            if mtr==6:
                #print('motif triangle')
                #print('This is the motif nod-2 {} and nod-1 {} nod {} and node-2 {} and node-1 {} and node {}'.format(nod-2,nod-1,nod,node-2,node-1,node))
                matrixm.append([nod-2,nod-1,nod,node-2,node-1,node])
    return matrixm

def nu_m(G,index_motifs):
    a_node=np.array(G.nodes())
    motif_nodes=[]
    for i in index_motifs:
        motif_nodes.append([a_node[i[0]],a_node[i[1]],a_node[i[2]]])
    
    num_m=0
    motif_n=[]
    for i in range(1,len(motif_nodes)):
        if motif_nodes[i-1]==motif_nodes[i]:
           num_m+=1 
        else: 
           motif_n.append([motif_nodes[i-1],num_m])
           num_m=0
    return motif_n  

pg = a_matrix(G, 1)
#L = nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()
adjac,degree,Lap=a_matrix(G,0)
index_motifs=mt_m(G,adjac)
mot_n=nu_m(G,index_motifs)
n = G.number_of_nodes()
k_max = max(G.degree())[1]#max degree 
I = np.matrix(np.identity(n))
m = (2 * k_max * I) - Lap
lev = leading_eigen_vector(m)
fv = eigenVec_2nd_smallest_eigenVal(lev, m)
#print(fv)
c1, c2 = get_partition(G, fv)
conductance=edge_conductance(c1, c2,G)
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
def km_feat(G,lev):
    import pandas as pd
    all_nodes=np.array(G.nodes())
    features=[]
    for node in range(len(all_nodes)):
        edges=len(list(G.neighbors(all_nodes[node])))
        features.append([edges,np.array(lev[node])[0][0]])
    features=np.array(features)
    datan=pd.DataFrame(features)
    datan.head()
    return datan
data=km_feat(G,lev) 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()


# defining the kmeans function with initialization as k-means++
algorithm = (KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(data_scaled)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_
pred=algorithm.predict(data_scaled)

com1=[]
com2=[]
for i in range(len(pred)):
    if pred[i]==0:
        com1.append(i)
    else:
        com2.append(i)

all_nodes=np.array(G.nodes())
c1k=[]
c2k=[]
for index in range(len(com1)-1):
    c1k.append(all_nodes[com1[index]])
for index in range(len(com2)-1):
    c2k.append(all_nodes[com2[index]])

def mpred(community1):        
    pred_eigen=[]
    all_nodes=np.array(G.nodes())
    all_nodes=all_nodes[:-1]
    for nodec in all_nodes:
        if nodec in community1:
            pred_eigen.append(int(0))
        else:
            pred_eigen.append(int(1))
    pred_e=np.array(pred_eigen)
    return pred_e

pred_e=mpred(c1)
pred_k=mpred(c1k)



tp=0
tn=0
for i in range(len(pred_e)):
    if pred_e[i]==pred_k[i]: #and pred_e[i]==0:
        tp+=1
    elif pred_e[i]!=pred_k[i]: #and pred_e[i]==1:
        tn+=1
precision=tp/(tp+tn)
print('This is precision {}'.format(precision))

#print('tp {} tn {} fp {} fn {}'.format(tp,tn,fp,fn))


from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(pred_e, pred_k)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(pred_e,pred_k))
print(classification_report(pred_e,pred_k))
