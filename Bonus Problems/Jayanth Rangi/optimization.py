
# coding: utf-8

# In[8]:


import pandas as pd
data=pd.read_excel('opt.xlsx')
print(data)


# In[11]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=8 , random_state=0).fit(data)


# In[14]:


data['label']=kmeans.labels_
print(data)


# In[17]:


import matplotlib.pyplot as plt
plt.scatter(data['marks'],data['label'])
plt.show()


# In[28]:


def grade_func(label):
    if label==2:
        return 10
    if label==7:
        return 9
    if label==0:
        return 8
    if label==6:
        return 7
    if label==4:
        return 6
    if label==5:
        return 5
    if label==1:
        return 4
    if label==3:
        return 'FR'
    
data['grade']=data['label'].apply(grade_func)
print(data)

# before using below code change 'FR' to 0 in function grade_func
'''plt.scatter(data['marks'],data['grade'])
plt.xlabel('MARKS')
plt.ylabel('GRADE')
plt.show()
'''    


# In[33]:


data[['marks','grade']].to_excel('grades.xlsx')

