
# coding: utf-8

# In[2]:


import numpy as np
import random as rd
import math as math
import pandas as pd

# initializing centroids(i.e. mean)
centroid=[95, 85, 75, 65, 55, 45, 25, 15]
centroid=np.reshape(np.asarray(centroid),(1,len(centroid)))

# using csv file for extracting column of total marks from excel sheet
names = ['total marks']
# put name of .csv file and give column number of total column in usecols
df_x = pd.read_csv( 'EE5327.csv', header=None, names=names, usecols=[18])
df_x = df_x[1:]
df_x = np.asarray(df_x)
total_marks = df_x
total_marks = total_marks.astype(float)

# this function will allot grade for given total marks
def grade(total_marks, points):
    g=[]
    maxi=np.argmax(total_marks,axis=0)
    for j in range(total_marks.shape[0]):
        if j == maxi:
            g = np.append(g,'A')
        else:
            w= points[j]
            if w==0:
                g=np.append(g,'A-')
            elif w==1:
                g=np.append(g,'B')
            elif w==2:
                g=np.append(g,'B-')
            elif w==3:
                g=np.append(g,'C')
            elif w==4:
                g=np.append(g,'C-')
            elif w==5:
                g=np.append(g,'D')
            elif w==6:
                g=np.append(g,'FS')
            elif w==7:
                g=np.append(g,'FR')
            else:
                pass
    return g

# this function do clustering
def group(total_marks, points):
    groups={}
    for k in range(centroid.shape[1]):
        groups[k]=[]
    
    for i in range(total_marks.shape[0]):
        
        w=points[i]
        marks= total_marks[i]
        
        if w==0:
            groups[0]= np.append(groups[0],marks)
            
        elif w==1:
            groups[1]= np.append(groups[1],marks)
            
        elif w==2:
            groups[2]= np.append(groups[2],marks)
            
        elif w==3:
            groups[3]= np.append(groups[3],marks)
            
        elif w==4:
            groups[4]= np.append(groups[4],marks)
            
        elif w==5:
            groups[5]= np.append(groups[5],marks)
            
        elif w==6:
            groups[6]= np.append(groups[6],marks)
            
        elif w==7:
            groups[7]= np.append(groups[7],marks)
            
        else:
            pass
    return groups

for l in range(7):# 5 times iteration
    new_centroid=np.zeros((centroid.shape))# store new centroid value
    
    # compute euclidean distance between total marks and centroid
    dist=total_marks-centroid
    dist=np.sqrt(dist**2)
    points=np.reshape(np.argmin(dist,axis=1),(len(total_marks),1))
    
    groups = group(total_marks, points)# clustering
    # computing new centroid
    for k in range(centroid.shape[1]):
        if len(groups[k])==0:
            groups[k]= [0]
            new_centroid[0][k]= centroid[0][k]
        else:
            new_centroid[0][k]= np.sum(groups[k])/len(groups[k])
    
    # compute error between centroid and new centroid
    error = np.linalg.norm(new_centroid-centroid,ord=2)
    centroid=new_centroid
    # giving labels(i.e. grades) to total marks
    grades=grade(total_marks,points)
    
grades= np.reshape(np.asarray(grades),(total_marks.shape))
grading=np.hstack((total_marks,grades))
print('total marks  grades')
print(grading)

