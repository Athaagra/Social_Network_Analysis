#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 05:00:08 2021

@author: kel
"""

import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
#import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
from math import sin, cos, sqrt, atan2, radians
import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from numpy import load
import time
import operator
import pandas as pd
import sklearn.mixture as mix
import scipy.stats as scs
from scipy.stats import multivariate_normal as mvn
import sklearn.mixture as mix
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



#TrainSet
#ROne=[4,4,4,5,1,3,5,2,3,5,2,1,2,3,4,5,1,2]
#RTwo=[5,5,2,3,5,4,2,1,5,1,2,5,3,2,3,1,4,4]
#Genre=['c','d','c','c','c','c','d','d','d','c','d','d','d','d','d','d','d','c']
#TestSet

os.chdir('/home/kel/Desktop/SocialNetworkAnalysis/')
fileE = pd.read_csv("flickrEdges_adj.tsv", sep='\t')
#read from file file
Title= fileE.columns
print(Title)
fileE = fileE.rename(columns={Title[0]:'Source',Title[1]:'Target',Title[2]:'Degree'})
Gs = nx.from_pandas_edgelist(fileE , source='Source', target='Target')
Ga = Gs.to_directed()
centrality = nx.eigenvector_centrality(Ga,max_iter=20)
sorted((v, f"{c:0.2f}") for v, c in centrality.items())
fileE['Eigenvalue']= np.nan
#for number in fileE['Source']:
#    value=centrality.get(number)
#    fileE['Eigenvalue'][np.where(fileE.Source==number)[0]]=value
#    fileE['Eigenvalue'] = np.where(fileE.Source==number, value,fileE['Eigenvalue'])
eigenvalues = pd.DataFrame(centrality.items()) 

n = len(fileE['Source'])
dataS= fileE['Source']
dataE= fileE['Eigenvalue']
dataD= fileE['Degree']
dataE=np.array(dataE).reshape((len(dataE),1))
dataS=np.array(dataS).reshape((len(dataS),1))
Data=np.hstack((dataE,dataS))
data = pd.DataFrame(Data, columns=['Eigen','Source'])
data=data.fillna(0)
dataEigen = np.array(data['Eigen'])
threshold=dataEigen.mean()
data['Th']=np.nan
#data['Th'][np.where(data.Eigen>threshold)[0]]='1'
#data['Th'][np.where(data.Eigen<threshold)[0]]='0'
#data['Th'].fillna('0')s
GFlick=[]
for i in dataS:
    GFlick.append([i[0],len(list(Ga.neighbors(i[0])))])
GFlickA=np.array(GFlick)


from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=2, random_state=0).fit(GFlickA)
kmean_labels = kmeans.labels_
kmeans.cluster_centers_



def normalize(a):
    return a/a.sum()

#E-step
def estep(h,jd,cod,featO,communities):
    EMgC=float(h[np.where(h[:,0]==communities[0])][0][1])*float(jd[np.where(jd[:,0]==communities[0])][0][1])*float(cod[np.where(cod[:,1]==str(featO))][0][2])#*float(CDt[np.where(CDt[:,1]==str(featT))][0][2])
    EMgD=float(h[np.where(h[:,0]==communities[0])][0][1])*float(jd[np.where(jd[:,0]==communities[1])][0][1])*float(cod[np.where(cod[:,1]==str(featO))][1][2])#*float(CDt[np.where(CDt[:,1]==str(featT))][1][2])
    #print(EMgC,EMgD)
    denominator=EMgC+EMgD
    NewEc=EMgC/denominator
    NewEd=EMgD/denominator
    NewJDc=NewEc+float(jd[np.where(jd[:,0]==communities[0])][0][1])
    NewJDd=NewEd+float(jd[np.where(jd[:,0]==communities[1])][0][1])
    return h,jd,cod,NewJDc,NewJDd,NewEc,NewEd

def Mstep(cod,featO,communities,NewEcp):
    #update parameters
    for i in range(len(cod[np.where(cod[:,1]==featO)])):
        if cod[np.where(cod[:,1]==featO)][i][0]==communities:
            cod[np.where(cod[:,1]==featO)[0][i]][2]=float(cod[np.where(cod[:,1]==featO)][i][2])+(NewEcp*float(featO))
    # normalize the data attributes
    reshaped = np.reshape(cod[np.where(cod[:,0]==communities)][:,2], (1, cod[np.where(cod[:,0]==communities)][:,2].shape[0]))
    reshaped = reshaped[0].astype(float)
    normalized = normalize(reshaped)
    for i in range(len(cod[np.where(cod[:,0]==communities)[0]][:,2])):
        cod[np.where(cod[:,0]==communities)[0][i]][2]=normalized[i]
    return cod

def trainingprocess(h,jd,cod,featO,ds):
    H,jdU,codI,JDc,JDd,NewExc,NewExd=estep(h,jd,cod,featO,ds)
    jdU[0][1]=JDc/2
    jdU[1][1]=JDd/2
    codD=Mstep(codI,featO,distinctElements[0],NewExc)
    codDU=Mstep(codD,featO,distinctElements[1],NewExd)
    return jdU,codDU,NewExc,NewExd

Genre=kmean_labels
Genre=[str(i) for i in Genre]
ROne=GFlickA[:,1]
gen=np.array(Genre)
rO=np.array(ROne)

#rT=np.array(RTwo)
distinctElements=np.unique(gen)
distinctElements=[str(i) for i in distinctElements]
H=[]
Jointdistribution=[]
ConditionaldistributionfOne=[]
ConditionaldistributionfTwo=[]
for cat in distinctElements:
    Jointdistr = len(np.where(gen==str(cat))[0])/len(Genre)
    countR=rO[np.where(gen==str(cat))]
    for count in countR:
        Conditionaldistrib = len(np.where(countR==count)[0])/len(countR)
        ConditionaldistributionfOne.append([cat,count,Conditionaldistrib])
    #countR=rT[np.where(gen==str(cat))]
    #for count in countR:
    #    Conditionaldistrib = len(np.where(countR==count)[0])/len(countR)
    #    ConditionaldistributionfTwo.append([cat,count,Conditionaldistrib])
    Jointdistribution.append([cat,float(Jointdistr)])
    H.append([cat,float(1/len(distinctElements))])
JD=np.array(Jointdistribution)
H=np.array(H)
CDo=np.array(ConditionaldistributionfOne)
CDo=np.unique(CDo,axis=0)
CDt=np.array(ConditionaldistributionfTwo)
CDt=np.unique(CDt,axis=0)

#JDc,JDd,NewEc,NewEd=estep('1','2',distinctElements)
#JD[0][1]=JDc/2
#JD[1][1]=JDd/2
#CDo=Mstep('1',distinctElements[0],NewEc)
#CDo=Mstep('2',distinctElements[1],NewEd)
#JDc,JDd,NewEc,NewEd=estep('3','1',distinctElements)
#JD[0][1]=JDc/2
#JD[1][1]=JDd/2
#CDo=Mstep('3',distinctElements[0],NewEc)
#CDo=Mstep('1',distinctElements[1],NewEd)
print(JD)
print(CDo)

  

 
Results=[]
plotting=[]
thplot=[]
iteration=0
while iteration != 1:
    count=0
    iteration+=1
    for j in range(len(ROne)):
        JDAU,CDoT,NewEC,NewED=trainingprocess(H,JD,CDo,ROne[j],distinctElements)
        label = CDoT[np.where(CDoT[:,1]==str(ROne[j]))]
        category = label[np.argmax(CDoT[np.where(CDoT[:,1]==str(ROne[j]))][:,2])]
        plotting.append([int(j),float(category[2])]) 
        thplot.append([int(j),NewEC,NewED])
        label = category[0]
        Results.append([Genre[j],str(label)])
        if str(label) ==Genre[j]:
            count+=1 
            #print(iteration)
        print('Iteration {} True Positive {} Correct Labeled {} '.format(iteration,count/len(ROne),count))

plt.figure(figsize=(18, 18))  
plotting=np.array(plotting)
with open('MaxEA.npy', 'wb') as f:
    np.save(f, plotting)
#plt.plot(plotting[:,0], plotting[:,1])
#plt.grid(True,which="both",ls="--",c='gray')
#L=plt.legend()
#L.get_texts().set_text('theta')
#plt.show()

plt.figure(figsize=(18, 18))  
thplot=np.array(thplot)
with open('MaxE.npy', 'wb') as f:
    np.save(f, thplot)
#plt.plot(thplot[:,0], thplot[:,1], 'red')
#plt.plot(thplot[:,0], thplot[:,2], 'green')
#plt.grid(True,which="both",ls="--",c='gray')
#L=plt.legend()
#L.get_texts().set_text('theta')
#plt.show()