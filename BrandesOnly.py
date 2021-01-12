from collections import deque
from numpy import load
import time
import numpy as np

# Decorator definition
def my_better_decorator(my_fn):
    # This function modifies the existing function
    def modified_function(*args, **kwargs):
        my_fn(*args, **kwargs)
    #Return the modified function
    return modified_function

@my_better_decorator
def brandes(V, A):
    C = dict((v,0) for v in V)
    for s in V:
        S = []
        P = dict((w,[]) for w in V)
        g = dict((t, 0) for t in V); g[s] = 1
        d = dict((t,-1) for t in V); d[s] = 0
        Q = deque([])
        Q.append(s)
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in A[v]:
                if d[w] < 0:
                    Q.append(w)
                    d[w] = d[v] + 1
                if d[w] == d[v] + 1:
                    g[w] = g[w] + g[v]
                    P[w].append(v)
                    print(P)
                    print(S)
        e = dict((v, 0) for v in V)
        while S:
            w = S.pop()
            for v in P[w]:
                e[v] = e[v] + (g[v]/g[w]) * (1 + e[w])
            #print(v,w,e[v])
            print(w,s)
            if w != s:
                C[w] = C[w] + e[w]
                print(C[w])
    return C

def Adjlist(Graph):
    A=[]
    for n, nbrs in Graph.adj.items():
       for nbr, eattr in nbrs.items():
           wt = eattr['weight']
           A.append((n,nbr,wt))
    return A


results=[]
titles = ['MusaeFacList.npy','LastfmAsiaList.npy','DeezerRlist.npy','Arxivlist.npy','USgridAdjlist.npy']
for title in titles:
    MusFace = load(title)
    MusFace = MusFace.astype(int)
    # print the array
    GraphAdjlist = MusFace[:10000, :3]
    musfac = np.unique(MusFace)
    startB = time.time()
    resultBrandes = brandes(musfac,GraphAdjlist)
    #end time
    endB = time.time()
    timeBrandes = endB - startB
    results.append(title,resultBrandes,timeBrandes)
np.save('ResultBrandes.npy', results)
