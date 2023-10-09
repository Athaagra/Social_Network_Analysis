#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 18:55:03 2023

@author: Optimus
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


def make_graph(G):
    # check if Graph is directed
    print('Directed:', nx.is_directed(G))

    # check if Graph is weighted
    print('Weighted:', nx.is_weighted(G))
    print()
    
    # converting to directed Graph for PageRank
    if not nx.is_directed(G):
        print('Graph converted to directed..')
        #G = G.to_directed() 
        G = G.to_undirected()

    print('Directed:', nx.is_directed(G))
    print()

    # labelling nodes as integers
    print('Relabelling nodes to integers..')
    n_unique_nodes = len(set(G.nodes()))
    node2int = dict(zip(set(G.nodes()), range(n_unique_nodes)))
    int2node = {v:k for k,v in node2int.items()}

    G = nx.relabel_nodes(G, node2int)

    # remove isolated nodes
    print('Removing isolated nodes..')
    nodes = G.nodes()
    for node in nodes:
        if len(G.edges(node))==0:
            G.remove_node(node)
    return G, int2node           

def plot_graph(G, final_probs, int2node, bool_final_probs=False):
    
    # defining labels
    labels = int2node

    # zachary karate club
    try:
        clubs = np.array(list(map(lambda x: G.nodes[x]['club'], G.nodes())))
        labels = dict(zip(G.nodes(), clubs)) 
    except:
        pass   

    if not bool_final_probs:
        nx.draw(G, with_labels=True, alpha=0.8, arrows=False, labels=labels)
    else:
        nx.draw(G, with_labels=True, alpha=0.8, arrows=False, node_color = final_probs, \
                                                                                        cmap=plt.get_cmap('viridis'), labels=labels)

        # adding color bar for pagerank importances
        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap('viridis'), norm=plt.Normalize(vmin = min(final_probs), vmax=max(final_probs)))
        sm._A = []
        plt.colorbar(sm)  
    return plt

def make_pagerank_matrix(G, alpha):
    n_nodes = len(G.nodes())

    # building adjacent matrix
    adj_matrix = np.zeros(shape=(n_nodes, n_nodes))
    for edge in G.edges():
        adj_matrix[edge[0], edge[1]] = 1

    # building transition probability matrix
    print('This is adjacency matrix {}'.format(adj_matrix))
    tran_matrix = 1# adj_matrix / np.sum(adj_matrix, axis=1).reshape(-1,1)

    # building random surfer matrix
    random_surf = np.ones(shape = (n_nodes, n_nodes)) / n_nodes    

    # building transition matrix for absorbing nodes
    absorbing_nodes = np.zeros(shape = (n_nodes,))
    for node in G.nodes():
        if len(G.edges(node))==0:
        #if len(G.out_edges(node))==0:
            absorbing_nodes[node] = 1
    absorbing_node_matrix = np.outer(absorbing_nodes, np.ones(shape = (n_nodes,))) / n_nodes

    # stochastic matrix
    stochastic_matrix = tran_matrix + absorbing_node_matrix

    # pagerank matrix
    pagerank_matrix = alpha * stochastic_matrix + (1-alpha) * random_surf
    return adj_matrix#pagerank_matrix

def random_walk(G, alpha, n_iter):
    n_nodes = len(G.nodes())
    initial_state = np.ones(shape=(n_nodes,)) / n_nodes
    pagerank_matrix = make_pagerank_matrix(G, alpha)

    new_initial_state = initial_state
    print('Running random walk..')
    NORM = []
    for i in range(n_iter):
        final_state = np.dot(np.transpose(pagerank_matrix), new_initial_state)
        
        prev_initial_state = new_initial_state
        new_initial_state = final_state
        L2 = np.linalg.norm(new_initial_state-prev_initial_state)
        NORM.append(L2)
        if np.allclose(new_initial_state, prev_initial_state):
            print(f'Converged at {i+1} iterations..')
            break

    plt.figure(figsize=(5,4))
    plt.plot(range(i+1), NORM)
    plt.xlabel('iterations')
    plt.ylabel('Euclidean Norm')
    plt.title('Convergence plot')
    plt.show()
    return final_state, pagerank_matrix

def run(G, alpha, n_iter):

    G, int2node = make_graph(G)
    print()
    print('Number of nodes: ', len(G.nodes()))
    print('Number of edges: ', len(G.edges())) 
    print()    

    final_probs,pagerank_matrix = random_walk(G, alpha, n_iter)

    # ensuring pagerank importance for each node
    assert len(final_probs) == len(G.nodes())

    # ensuring probabilities sum to 1
    print('This is pagerank matrix {}'.format(pagerank_matrix))
    assert np.allclose(np.sum(final_probs), 1)

    print()
    print('Pagerank importances..')
    print(final_probs)
    print('This is pagerank matrix {}'.format(pagerank_matrix))

    plt.figure(figsize=(25,8))
    plt.subplot(121)
    plot_graph(G, None, int2node, bool_final_probs=False)
    plt.subplot(122)
    plot_graph(G, final_probs, int2node, bool_final_probs=True)
    plt.show()
    return final_probs,pagerank_matrix

alpha = 0.8
n_iter = 1000

G = nx.karate_club_graph()
final_probs,pagerank_matrix = run(G, alpha, n_iter)

pg = make_pagerank_matrix(G, 1)

matrixm=[]
for nod in range (2,len(G.nodes)):
    for node in range (2,len(G.nodes)):
        Tv=np.vstack((pg[nod-2][node-2:node+1],pg[nod-1][node-2:node+1]))
        Tre=np.vstack((Tv,pg[nod][node-2:node+1]))
        mtr=sum(sum(Tre))
        if mtr==6:
            print('motif triangle')
            print('This is the motif {} and node {} nod {}'.format(Tre,node,nod))
        matrixm.append(Tre)

#Conductance and clustering measures
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
    c11 = set(i+1 for i in c[:16])#pick 1st 16
    c12 = set(j+1 for j in c[16:])# pick rest i.e 18
    #partition 2
    c21 = set(j+1 for j in c[-16:])#pick last 16
    c22 = set(j+1 for j in c[:-16])#pick rest
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





def a_matrix(G, alpha):
    n_nodes = len(G.nodes())
    edges = G.edges()
    # building adjacent matrix
    adj_matrix = np.zeros(shape=(n_nodes, n_nodes))
    for edge in G.edges():
        adj_matrix[edge[0], edge[1]] = 1
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
        degree_matrix[i[0],i[0]]=i[1]
    L=adj_matrix+degree_matrix
    return adj_matrix,degree_matrix,L

def edge_conductance(c1,c2):
    c1=list(c1)
    c2=list(c2)
    edges = G.edges()
    ed=list(edges)
    shared_con=0
    for q in range(len(c1)):
        for x in range(len(c2)):
            if (c1[q],c2[x]) in ed:
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