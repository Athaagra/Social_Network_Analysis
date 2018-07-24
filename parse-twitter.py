# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 10:14:57 2017

@author: Kel3vra 
"""

import re
from collections import Counter
from datetime import datetime
from time import mktime
import csv


#import date_converter
#"(^|[^@\w])@(\w{1,15})"
#r"@\w*"
#"@([a-zA-Z0-9]{1,15})"
mt_dict={}
R=[]
#load the file and clean the data
with open('twitter-larger.in','r',encoding="utf8") as f:
    for line in f:
        linet=line.split('\t')
        user=linet[1]
        time= linet[0]
        #finde the username
        mention= re.findall(r"@(\w+)", (linet[2]))
        md=mention,time
        #put the data in a list
        R.append([user,mention,time])
        #put data in a dict
        if user not in mt_dict:
            mt_dict[user] = []
        if time not in mt_dict[user]:
            mt_dict[user].append([mention,time])
       
r=[]
r1=[]
r2=[]
#reading the list and clean empty usernames
for x in range(len(R)):
    times= R[x][2]
    user=R[x][0]
    comment = R[x][1]
    if comment==[]:
     r1.append([user,comment,times])
    count=0
    #separate username and mentios for every user
    while len(comment)>count:
        times= R[x][2]
        user=R[x][0]
        co=comment[count]
        r2.append([user,co,times])
        count = count + 1

#count the element user and mention
c = Counter((elem[0],elem[1])for elem in r2)

#create a dictionary and convert time
data_dict1 = []
for x in r2:
    times= x[2]
    times = mktime(datetime.strptime(times, "%Y-%m-%d %H:%M:%S").timetuple())
    for k,v in (c.items()):
            if k[0]==x[0] and k[1]==x[1]:
                data_dict1.append([x[0],x[1],v,times])
           
#sorting the data      
data_dict1.sort()
#make the user
data_dict6 = {}
for x in data_dict1:
    key = x[0],x[1],x[2]
    data_dict6.setdefault(key,[]).append([x[3]])
#create the adj list convert the time to sting again
adj_list={}
for k,v in data_dict6.items():
    q = v[0][0]
    date = datetime.fromtimestamp(q).strftime("%Y-%m-%d %H:%M:%S")
    [k[1],k[2],date]
    if k[0] not in adj_list:
            adj_list[k[0]] = []
    v=[k[1],k[2],date]
    if v not in  adj_list[k[0]]:
             adj_list[k[0]].append(v)
#create the csv 
with open('twitter_large.csv', 'w',encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file, delimiter='\t',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for key, values in adj_list.items():
       for value in values:
        writer.writerow([key]+value)
        

