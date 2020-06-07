#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import time

warnings.filterwarnings("ignore")

os.chdir(r"C:\Users\ezkiska\Videos\Imarticus\Python\6th Week 11th & 12th Jan\SAT 11th Jan Kmeans clustering")

data = pd.read_csv('snsdata.csv')
data.head()


# In[2]:


data.describe()


# In[3]:


data.describe(include = 'O')


# In[4]:


data.dtypes


# In[5]:


data.isnull().sum()


# In[6]:


'''gender and age has missing values'''

# gender is categorical
data.gender.value_counts()


# In[7]:


data.gender.value_counts(dropna = False)
data.gender.value_counts(dropna = False)/data.shape[0] # will return proportion


# In[9]:


'''
# Here, we see that 2,724 records (9 percent) have missing gender data. Interestingly, there are over
# four times as many females as males in the SNS data, suggesting that males are not as inclined to 
# use SNS websites as females.
'''
# age is continuous
data.age.describe()


# In[11]:


data.age.isnull().sum()


# In[12]:


data.age.isnull().sum()/data.shape[0] #percentage calculatio approx 17 percent.


# In[13]:


sns.distplot(data.age.fillna(data.age.median()))
data.age.max() #106.927


# In[14]:


data.age.min() #3.0860000000000003

'''
# A total of 5,086 records (17 percent) have missing ages. Also concerning is the fact that the
# minimum and maximum values seem to be unreasonable; it is unlikely that a 3 year old or a 106 year 
# old is attending high school. To ensure that these extreme values don’t cause problems for the 
# analysis, we’ll need to clean them up before moving on.

# A more reasonable range of ages for the high school students includes those who are at least 13 
# years old and not yet 20 years old. Any age value falling outside this range should be treated 
# the same as missing data-we cannot trust the age provided. To recode the age variable, we can use 
# the ifelse() function, assigning teenagethevalueofteenage if the age is at least 13 and less than 
# 20 years; otherwise, it will receive the value NA:
'''
# In[16]:


data.loc[(data.age < 13), 'age'] = np.nan


# In[17]:


data.loc[(data.age >= 20), 'age'] = np.nan


# In[18]:


'''
# By rechecking the summary() output, we see that the age range now follows a distribution that 
# looks much more like an actual high school:
'''

data.age.isnull().sum()


# In[19]:


data.age.describe()


# In[20]:


data.age.isnull().sum()/data.shape[0]


# In[21]:


'''
# Unfortunately, now we’ve created an even larger missing data problem. We’ll need to find a way to deal with these values before continuing with our analysis.

# Data preparation - dummy coding missing values
'''
data['gender'] = data.gender.fillna('Unknown')
data.gender.value_counts()


# In[22]:


df_gender = pd.get_dummies(data.gender, drop_first = 'True') #Convert categorical variable into dummy/indicator variables.
'''#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html'''

''' Data preparation - imputing the missing values
'''


# In[23]:


np.mean(data.age)


# In[24]:


data[['age', 'gradyear']].groupby(['gradyear'], as_index=False).mean().sort_values(by='gradyear', ascending=True)

data.loc[(data.age.isnull()) & (data.gradyear == 2006), 'age'] = 18.656
data.loc[(data.age.isnull()) & (data.gradyear == 2007), 'age'] = 17.706
data.loc[(data.age.isnull()) & (data.gradyear == 2008), 'age'] = 16.768
data.loc[(data.age.isnull()) & (data.gradyear == 2009), 'age'] = 15.819


# In[25]:


data.age.isnull().any()


# In[26]:


data1 = data.drop(['gender'], axis = 1)
df = pd.concat([data1, df_gender], axis = 1)


# In[27]:


data = df.copy()


# In[28]:


from sklearn import preprocessing

## scaling
scaler = preprocessing.StandardScaler().fit(df[['age', 'friends']])
dfs = scaler.transform(df[['age', 'friends']])

df[['age', 'friends']] = dfs

df = df.drop(['gradyear', 'M', 'Unknown'], axis = 1)

data = df.copy()


# In[29]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
data['cluster'] = kmeans.labels_ #model.predict(df)
data['cluster'].value_counts()

index_remove = data['cluster'][data['cluster'] == 2].index[0]
df =  df.drop(df.index[[index_remove]])
data = data.drop(data.index[[index_remove]])

kmeans = KMeans(n_clusters=3, random_state=0).fit(df)
data['cluster'] = kmeans.labels_ #model.predict(df)
data['cluster'].value_counts()
centers = kmeans.cluster_centers_
cluster_assignments = kmeans.labels_
X = df.values
np.sum((X - centers[cluster_assignments]) ** 2)

sil = []


# In[30]:


# Use silhouette coefficient to determine the best number of clusters
from sklearn.metrics import silhouette_score

for n_cluster in  list(range(2,11)): #[4,5,6,7,8]:
    kmeans = KMeans(n_clusters=n_cluster).fit(df)
    
    silhouette_avg = silhouette_score(df, kmeans.labels_)
    
    print('Silhouette Score for %i Clusters: %0.4f' % (n_cluster, silhouette_avg))
    
    sil.append(silhouette_avg)


krange = list(range(2,11))
plt.plot(krange, sil)
plt.xlabel("$K$")
plt.ylabel("Silhoutte score")
plt.show()


# In[31]:


from sklearn import cluster
import numpy as np

sse = []
krange = list(range(2,11))
X = df.values
for n in krange:
    model = cluster.KMeans(n_clusters=n, random_state=3)
    model.fit_predict(X)
    cluster_assignments = model.labels_
    centers = model.cluster_centers_
    sse.append(np.sum((X - centers[cluster_assignments]) ** 2))

plt.plot(krange, sse)
plt.xlabel("$K$")
plt.ylabel("Sum of Squares")
plt.show()


# In[32]:


# 3 optimal clusters


# using inertia
inertia = []
krange = list(range(2,11))
X = df.values
for n in krange:
    model = cluster.KMeans(n_clusters=n, random_state=3)
    model.fit_predict(X)
    labels = model.labels_
    inertia_ = model.inertia_
    inertia.append(inertia_)
      

plt.plot(krange, inertia)
plt.xlabel("$K$")
plt.ylabel("Inertia")
plt.show()


# In[33]:


# check groups

kmeans = KMeans(n_clusters=3, max_iter = 1000).fit(df)

data['cluster'] = kmeans.labels_ #model.predict(df)
data['cluster'].value_counts()


# In[ ]:




