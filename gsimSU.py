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

#E-step
def estep(h,jd,cod,featO,communities):
    EMgC=float(h[np.where(h[:,0]==communities[0])][0][1])*float(jd[np.where(jd[:,0]==communities[0])][0][1])*float(cod[np.where(cod[:,1]==featO)][0][2])
    EMgD=float(h[np.where(h[:,0]==communities[0])][0][1])*float(jd[np.where(jd[:,0]==communities[1])][0][1])*float(cod[np.where(cod[:,1]==featO)][1][2])
    print(featO,EMgC,EMgD)
    denominator=EMgC+EMgD+0.001
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

def similar(clusterr,kmean_labelss):
    countss=0
    clusterr=np.array(clusterr)
    #print(len(clusterr))
    #print('Kmean labelss {}'.format(len(kmean_labelss)))
    #print(clusterr.shape)
    for labell in range(len(clusterr)):
        #print(clusterr[labell])
        #print(kmean_labelss[labell])
        if clusterr[labell][1]==kmean_labelss[labell]:
            countss+=1
    print('Precision {}'.format(countss/len(clusterr)))

def CondiInd(dataSE,GaE,GenreE,distinctCatE):
    ClO=[]
    ClT=[]
    for x in range(len(dataS)):
        if GenreE[x]==distinctCat[0]:
            ClO.append([dataS[x][0]+list(Ga.neighbors(dataS[x][0]))])
        else:
            ClT.append([dataS[x][0]+list(Ga.neighbors(dataS[x][0]))])
    ClO=np.array(ClO)
    print(ClO.shape)
    ClON=[ClO[:,0][c][0] for c in range(len(ClO[:,0]))]
    ClT=np.array(ClT)
    ClTN=[ClT[:,0][v][0] for v in range(len(ClT[:,0]))]
    ClTN=np.array(ClTN)
    ClON=np.array(ClON)

    CountTrue=0
    for s in ClON:
        if s in ClTN:
            CountTrue+=1
    CI = (CountTrue/len(ClON))*(CountTrue/len(ClTN))
    print('CI {}'.format(CI))

def normalize(a):
    return a/a.sum()

datasets="loc-brightkite_edges_adj.tsv"
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
for q in dataS:
   # print('dataS {}'.format(i))
    GFlick.append([q[0],len(list(Ga.neighbors(q[0])))])
GFlickA=np.array(GFlick)
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=2, random_state=0).fit(GFlickA)
kmean_labels = kmeans.labels_
kmeans.cluster_centers_
distinctCat=np.unique(kmean_labels)
GenreI=kmean_labels
print('Kmeans:')
CondiInd(dataS,Ga,GenreI,distinctCat)
#Genre=[str(i) for i in GenreI]
#np.save('dataS.npy',dataS)
#np.save('Ga.npy',Ga)
#np.save('Genre.npy',Genre)
#np.save('distinctCat.npy',distinctCat)
#CondiInd(dataS,Ga,Genre,distinctCat)

#dataS=np.load('dataS.npy')
#Ga=np.load('Ga.npy')
#Genre=np.load('Genre.npy')
#distinctCat=np.load('distinctCat.npy')

distinctElements=np.unique(Genre)
#distinctElements=[str(i) for i in distinctElements]
H=[]
ROne=GFlickA[:,1]
gen=np.array(GenreI)
rO=np.array(ROne)
Jointdistribution=[]
ConditionaldistributionfOne=[]
for cat in distinctElements:
    Jointdistr = len(np.where(gen==cat)[0])/len(Genre)
    countR=rO[np.where(gen==cat)]
    for count in countR:
        Conditionaldistrib = len(np.where(countR==count)[0])/len(countR)
        ConditionaldistributionfOne.append([cat,count,Conditionaldistrib])
    Jointdistribution.append([cat,float(Jointdistr)])
    H.append([cat,float(1/len(distinctElements))])
JDi=np.array(Jointdistribution)
H=np.array(H)
CDo=np.array(ConditionaldistributionfOne)
CDo=np.unique(CDo,axis=0)



sigma=np.std(ROne)
ROne=ROne
gen=np.array(GenreI)
rO=np.array(ROne)
mValues=np.array([0.0,0.0])
mValues[0]=rO.mean()
mValues[1]=rO.mean()+1.5
sigma=np.array([0.0,0.0])
sigma[0]=rO.std()
sigma[1]=rO.std()*1.5
ROne=ROne
gen=np.array(GenreI)
rO=np.array(ROne)
we=np.array([0.2,0.2])
gamma=np.array([0.0,0.0])
LmO=[]
LmT=[]
densityFunction=[]
likelihood=[]
Results=[]
plotting=[]
thplot=[]
iteration=0
while iteration <23:
    cluster=[]
    count=0
    iteration+=1
    for i in rO:
	    JDAU,CDoT,NewEC,NewED=trainingprocess(H,JDi,CDo,i,distinctElements)
        label = CDoT[np.where(CDoT[:,1]==ROne[i])]
        category = label[np.argmax(CDoT[np.where(CDoT[:,1]==ROne[i])][:,2])]
        plotting.append([int(i),category[2]]) 
        thplot.append([int(i),NewEC,NewED])
        label = category[0]
        Results.append([GenreI[i],label])
        if label ==Genre[i]:
            count+=1 
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
           # print('gamma1 {} weight1 {} mvalue1 {} sigma1 {}'.format(gamma[0],we[0],mValues[0],sigma[0]))
            densityFunction.append([int(count),round(Lot, 4)])
        else:
            #print('The value {} value of Dis Two {} cluster Two'.format(i,EzT))
            cluster.append([i,distinctCat[1]])
            gamma[1]+=Ltt
            we[1]+=gamma[1]/len(rO)
            mValues[1]=(float(EnumeT/EzT))
            sigma+=Ltt*((i-mValues)*(i-mValues.transpose()))/gamma[1]
           # print('gamma2 {} weight2 {} mvalue2 {} sigma2 {}'.format(gamma[1],we[1],mValues[1],sigma[1]))
            densityFunction.append([int(count),round(Ltt, 4)])
            #mValues=normalize(mValues)
            #gamma=normalize(gamma)
        count+=1
    print('Iteration {},precsion,CI E-m'.format(iteration))
    similar(cluster,kmean_labels)
    cluster=np.array(cluster)
    cluEm=cluster[:,1]
    CondiInd(dataS,Ga,cluEm,distinctCat)
	print('Iteration {},precsion,CI E-m Supervised'.format(iteration))
    similar(Results,kmean_labels)
    result=np.array(Results)
    resuEm=result[:,1]
    CondiInd(dataS,Ga,resuEm,distinctCat)

        #precision(kmean_labels,cluster)
        #for labells in range(len(kmean_labels)):
        #  if kmean_labels[labells]==cluster[labells]:
        #     prc+=1
        #print('Precision {}'.format(prc/len(kmean_labels))
print(gamma)