#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:17:20 2021

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


    #start = 0.01
#mvalueO=rO.mean()
#mvalueT=rT.mean()
#start = 0.01
#stop = 0.9
#step = 0.001
#num = 1+int((stop-start)/step)
#np.linspace(start, stop, num, endpoint=True)
#gamma=np.arange(0.01,0.01,len(ROne))

#TrainSet
#ROne=[4,4,4,5,1,3,5,2,3,5,2,1,2,3,4,5,1,2]
#RTwo=[5,5,2,3,5,4,2,1,5,1,2,5,3,2,3,1,4,4]
#Genre=['c','d','c','c','c','c','d','d','d','c','d','d','d','d','d','d','d','c']

#sigma=np.std(ROne)
#Genre=Genre
#Genre=[str(i) for i in Genre]


def CondiInd(dataSE,GaE,GenreE,distinctCatE):
    ClO=[]
    ClT=[]
    for i in range(len(dataS)):
        if Genre[i]==str(distinctCat[0]):
            ClO.append([dataS[i][0]+list(Ga.neighbors(dataS[i][0]))])
        else:
            ClT.append([dataS[i][0]+list(Ga.neighbors(dataS[i][0]))])
    ClO=np.array(ClO)
    ClON=[ClO[:,0][i][0] for i in range(len(ClO[:,0]))]
    ClT=np.array(ClT)
    ClTN=[ClT[:,0][i][0] for i in range(len(ClT[:,0]))]
    ClTN=np.array(ClTN)
    ClON=np.array(ClON)

    CountTrue=0
    for i in ClON:
        if i in ClTN==True:
            CountTrue+=1
    CI = (CountTrue/len(ClON))*(CountTrue/len(ClTN))
    print('CI {}'.format(CI))

def normalize(a):
    return a/a.sum()

datasets="roadNet-PA_adj.tsv"
fileE = pd.read_csv(datasets, sep='\t')
#read from file file
Title= fileE.columns
print(Title)
fileE = fileE.rename(columns={Title[0]:'Source',Title[1]:'Target',Title[2]:'Degree'})
Gs = nx.from_pandas_edgelist(fileE , source='Source', target='Target')
Ga = Gs.to_directed()
n = len(fileE['Source'])
dataS= fileE['Source']
dataS=dataS.fillna(0)
dataS=np.array(dataS).reshape((len(dataS),1))
GFlick=[]
for i in dataS:
    GFlick.append([i[0],len(list(Ga.neighbors(i[0])))])
GFlickA=np.array(GFlick)
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=2, random_state=0).fit(GFlickA)
kmean_labels = kmeans.labels_
kmeans.cluster_centers_
distinctCat=np.unique(kmean_labels)
Genre=kmean_labels
Genre=[str(i) for i in Genre]
#np.save('dataS.npy',dataS)
#np.save('Ga.npy',Ga)
#np.save('Genre.npy',Genre)
#np.save('distinctCat.npy',distinctCat)
#CondiInd(dataS,Ga,Genre,distinctCat)

#dataS=np.load('dataS.npy')
#Ga=np.load('Ga.npy')
#Genre=np.load('Genre.npy')
#distinctCat=np.load('distinctCat.npy')


ROne=GFlickA[:,1]
sigma=np.std(ROne)
ROne=ROne
gen=np.array(Genre)
rO=np.array(ROne)
mValues=np.array([0.0,0.0])
mValues[0]=rO.mean()
mValues[1]=rO.mean()+1.5
sigma=np.array([0.0,0.0])
sigma[0]=rO.std()
sigma[1]=rO.std()*1.5
ROne=ROne
gen=np.array(Genre)
rO=np.array(ROne)
gamma=np.array([0.0,0.0])
we=np.array([0.2,0.2])
LmO=[]
LmT=[]
densityFunction=[]
likelihood=[]
cluster=[]
iteration=0
while iteration <23:
    count=0
    iteration+=1
    for i in rO:
        Lmo=np.exp(-np.square(i-mValues[0])/np.square(2*sigma[0]))
        Lmt=np.exp(-np.square(i-mValues[1])/np.square(2*sigma[1]))
        #Lo=1/sqrt(2*np.pi*np.square(sigma[0]))*Lmo
        #Lt=1/sqrt(2*np.pi*np.square(sigma[1]))*Lmt
        Lot=1/(np.square(we[0])*sqrt(np.pi*np.square(sigma[0])))*(Lmo+Lmt)
        Ltt=1/(np.square(we[1])*sqrt(np.pi*np.square(sigma[1])))*(Lmt+Lmo)
        likelihood.append([count,Lot])
        denomzo=Lmo+Lmt
        EzO=Lmo/denomzo
        denomzt=Lmt+Lmo
        EzT=Lmt/denomzt
        EnumeO=(float(i)*EzO)
        EnumeT=(float(i)*EzT)
        if EnumeO > EnumeT:
            #print('The value {} value of Dis One {} cluster One'.format(i,EzO))
            cluster.append([i,distinctCat[0]])
            gamma[0]+=Lot
            we[0]+=gamma[0]/len(rO)
            mValues[0]=(float(EnumeO/EzO))
            sigma+=Lot*((i-mValues)*(i-mValues.transpose()))/gamma[0]
        #print('gamma1 {} weight1 {} mvalue1 {} sigma1 {}'.format(gamma[0],we[0],mValues[0],sigma[0]))
            densityFunction.append([int(count),round(Lot, 4)])
        else:
            #print('The value {} value of Dis Two {} cluster Two'.format(i,EzT))
            cluster.append([i,distinctCat[1]])
            gamma[1]+=Ltt
            we[1]+=gamma[1]/len(rO)
            mValues[1]=(float(EnumeT/EzT))
            sigma+=Ltt*((i-mValues)*(i-mValues.transpose()))/gamma[1]
            #print('gamma2 {} weight2 {} mvalue2 {} sigma2 {}'.format(gamma[1],we[1],mValues[1],sigma[1]))
            densityFunction.append([int(count),round(Ltt, 4)])
            #mValues=normalize(mValues)
            #gamma=normalize(gamma)
        count+=1
print(gamma)


countss=0
for label in range(len(cluster)):
    if cluster[label][1]==kmean_labels[label]:
        countss+=1
print('Precision {}'.format(countss/len(cluster)))    
        
rra = np.arange(0,len(rO))
plt.figure(figsize=(18, 18))  
plotting=np.array(densityFunction)
#with open('MaxEA.npy', 'wb') as f:
#    np.save(f, plotting)
#plt.plot(rra,plotting[:,1], color='red')
#plt.plot(plotting[:,1], plotting[:,2], color='green')
#plt.grid(True,which="both",ls="--",c='gray')
#L=plt.legend()
#L.get_texts().set_text('theta')
#plt.show()


#both distribution in one diagram
plt.loglog(plotting[:,1], 'g',marker='o',label='likelihood')
plt.grid(True,which="both",ls="--",c='gray')
#plt.loglog(odegreesequence,'r',marker='<',label='OutDegree')
#title of diagram
plt.title("Density function")
#label y title
plt.ylabel("nodes")
#label x title
plt.xlabel("degree")
#save the figure
plt.savefig("DensityFunction.png")
#display the figure
plt.legend(loc="upper right")
plt.show()

ClO=[]
ClT=[]
for i in range(len(dataS)):
    if Genre[i]==str(distinctCat[0]):
        ClO.append([dataS[i][0]+list(Ga.neighbors(dataS[i][0]))])
    else:
       ClT.append([dataS[i][0]+list(Ga.neighbors(dataS[i][0]))])
ClO=np.array(ClO)
ClON=[ClO[:,0][i][0] for i in range(len(ClO[:,0]))]
ClT=np.array(ClT)
ClTN=[ClT[:,0][i][0] for i in range(len(ClT[:,0]))]
ClTN=np.array(ClTN)
ClON=np.array(ClON)
CountTrue=0
for i in ClON:
   if i in ClTN:
      print(i)
      CountTrue+=1
      print(CountTrue)
   else:
       print('false',i)
CI = (CountTrue/len(ClON))*(CountTrue/len(ClTN))
print('CI {}'.format(CI))


