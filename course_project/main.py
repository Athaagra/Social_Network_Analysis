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
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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
###################
#   deanonymize neighborsOne-Two
##################
#def normalize(a):
#    return a/a.sum()
nodes=G.nodes()
#nodes=list(nodes)
fnei=[]
fc=[]
for n in nodes:
    if n!=' Target' and n!='Source ':
        neig=list(G.neighbors(n))
        neigneig=[len(list(G.neighbors(ne))) for ne in neig]
        neigneig=[newneig for newneig in neigneig]
        degree=G.degree(n)
        degreeT=[G.degree(ne) for ne in neig]
        degreeT=np.array(degreeT)
        degreeT=np.sum(degreeT)
        fc.append([n,degree,len(neig),degreeT,len(neigneig)])
        fnei.append([n,degree,len(neig),neig,degreeT,len(neigneig),neigneig])
fnei=np.array(fnei)
fc=np.array(fc)
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
c=Counter(fnei[:,5])
counterO=0
counterT=0
counterTh=0
counterF=0
counterFi=0
OneFo=[]
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
OneFo.append([counterO/len(fnei),counterT/len(fnei),counterTh/len(fnei),counterF/len(fnei),counterFi/len(fnei)])
OneFi = pd.DataFrame(fc)
clustering = DBSCAN(eps=85, min_samples=4).fit(fc)
DBSCAN_dataset = OneFi.copy()
DBSCAN_dataset.loc[:,'Cluster'] = clustering.labels_
OneFi=DBSCAN_dataset.Cluster.value_counts().to_frame()
print(OneN,OneT,OneTh,OneFo,OneFi,file=open('queries.txt','w'))

# we use Pandas to work with the data as it makes working with categorical data very easy
import pandas as pd

# this is a list of the column names in our dataset (as the file doesn't contain any headers)
#names = (
#    'age',
#    'workclass', #Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#    'fnlwgt', # "weight" of that person in the dataset (i.e. how many people does that person represent) -> https://www.kansascityfed.org/research/datamuseum/cps/coreinfo/keyconcepts/weights
#    'education',
#    'education-num',
#    'marital-status',
#    'occupation',
#    'relationship',
#    'race',
#    'sex',
#    'capital-gain',
#    'capital-loss',
#    'hours-per-week',
#    'native-country',
#    'income',
#)

# some fields are categorical and will require special treatment
categorical = set((
    'Degree',
    'Neigh',
    #'workclass',
    #'education',
    #'marital-status',
    #'occupation',
    #'relationship',
    #'sex',
    #'native-country',
    #'race',
    #'income',
))
#df = pd.read_csv("adult.all.txt", sep=", ", header=None, names=names, index_col=False, engine='python');# We load the data using Pandas
#df.head()
#df=df[0:6]
OneFp=pd.DataFrame()
OneFp['Degree']=pd.DataFrame(fnei[0:,1])
OneFp['Neigh']=pd.DataFrame(fnei[1:,2])
df=OneFp
#for name in categorical:
#    df[name] = df[name].astype('category')

def get_spans(df, partition, scale=None):
    """
    :param        df: the dataframe for which to calculate the spans
    :param partition: the partition for which to calculate the spans
    :param     scale: if given, the spans of each column will be divided
                      by the value in `scale` for that column
    :        returns: The spans of all columns in the partition
    """
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans

full_spans = get_spans(df, df.index)
full_spans

def split(df, partition, column):
    """
    :param        df: The dataframe to split
    :param partition: The partition to split
    :param    column: The column along which to split
    :        returns: A tuple containing a split of the original partition
    """
    dfp = df[column][partition]
    print(dfp)
    if column in categorical:
        values = dfp.unique()
        print(values)
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)

def is_k_anonymous(df, partition, sensitive_column, k=3):
    """
    :param               df: The dataframe on which to check the partition.
    :param        partition: The partition of the dataframe to check.
    :param sensitive_column: The name of the sensitive column
    :param                k: The desired k
    :returns               : True if the partition is valid according to our k-anonymity criteria, False otherwise.
    """
    if len(partition) < k:
        return False
    return True

def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    """
    :param               df: The dataframe to be partitioned.
    :param  feature_columns: A list of column names along which to partition the dataset.
    :param sensitive_column: The name of the sensitive column (to be passed on to the `is_valid` function)
    :param            scale: The column spans as generated before.
    :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
    :returns               : A list of valid partitions that cover the entire dataframe.
    """
    finished_partitions = []
    partitions = [df.index]
    print(partitions)
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions

# we apply our partitioning method to two columns of our dataset, using "income" as the sensitive attribute
feature_columns = ['Degree', 'Neigh']
sensitive_column = 'income'
finished_partitions = partition_dataset(df, feature_columns, sensitive_column, full_spans, is_k_anonymous)
# we get the number of partitions that were created
len(finished_partitions)

import matplotlib.pylab as pl
import matplotlib.patches as patches

def build_indexes(df):
    indexes = {}
    for column in categorical:
        print(column)
        values = sorted(df[column].unique())
        print(values)
        indexes[column] = { x : y for x, y in zip(values, range(len(values)))}
    return indexes

def get_coords(df, column, partition, indexes, offset=0.1):
    if column in categorical:
        print(column)
        sv = df[column][partition].sort_values()
        print('This is the sorted values {}'.format(sv))
        l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]]+1.0
        print('This is the l,r {},{}'.format(l,r))
    else:
        sv = df[column][partition].sort_values()
        print('This is the sorted values {}'.format(sv))
        next_value = sv[sv.index[-1]]
        print('This is the next value {}'.format(next_value))
        larger_values = df[df[column] > next_value][column]
        print(larger_values)
        if len(larger_values) > 0:
            next_value = larger_values.min()
            print(next_value)
        l = sv[sv.index[0]]
        print(l)
        r = next_value
        print('This is the next_value {}'.format(r))
    # we add some offset to make the partitions more easily visible
    l -= offset
    print('subtract this parition to the offset {}'.format(l))
    r += offset
    print('add this parition to the offset {}'.format(r))
    return l, r

def get_partition_rects(df, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]):
    rects = []
    for partition in partitions:
        print(partition)
        print('This is the index {}'.format(indexes))
        print(column_x)
        print(column_y)
        xl, xr = get_coords(df, column_x, partition, indexes, offset=offsets[0])
        print(xl,xr)
        yl, yr = get_coords(df, column_y, partition, indexes, offset=offsets[1])
        print(yl,yr)
        rects.append(((xl, yl),(xr, yr)))
    return rects

def get_bounds(df, column, indexes, offset=1.0):
    if column in categorical:
        return 0-offset, len(indexes[column])+offset
    return df[column].min()-offset, df[column].max()+offset

# we calculate the bounding rects of all partitions that we created
feature_columns=['Degree','Neigh']
indexes = build_indexes(df)
column_x, column_y = feature_columns[:2]
print(column_x)
print(column_y)
rects = get_partition_rects(df, finished_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])
# let's see how our rects look like
rects[:10]

# we plot the rects
def plot_rects(df, ax, rects, column_x, column_y, edgecolor='black', facecolor='none'):
    for (xl, yl),(xr, yr) in rects:
        ax.add_patch(patches.Rectangle((xl,yl),xr-xl,yr-yl,linewidth=1,edgecolor=edgecolor,facecolor=facecolor, alpha=0.5))
    ax.set_xlim(*get_bounds(df, column_x, indexes))
    ax.set_ylim(*get_bounds(df, column_y, indexes))
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)

pl.figure(figsize=(20,20))
ax = pl.subplot(111)
plot_rects(df, ax, rects, column_x, column_y, facecolor='r')
pl.scatter(df[column_x], df[column_y])
pl.show()
