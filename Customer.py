#!/usr/bin/env python
# coding: utf-8

# In[292]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
import pylab
from sklearn.cluster import DBSCAN


# In[293]:


df = pd.read_csv('Customer.csv')
df
#print(df.dtypes)


# In[294]:


DF = df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
DF


# In[295]:


from sklearn import preprocessing
le_gender= preprocessing.LabelEncoder()
le_gender.fit(['Female', 'Male'])
DF[:,1] = le_gender.transform(DF[:,1])
DF


# In[296]:


X = DF[:,1:]
X


# # K-Mean

# In[297]:


from sklearn.cluster import KMeans
clustnum = 4
k_means = KMeans(n_clusters=clustnum, init='k-means++', n_init=12, max_iter=300,
                 tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[298]:


df['Labels (K-Mean)'] = labels
df


# In[299]:


df.groupby('Labels (K-Mean)').mean() 


# In[300]:


plt.scatter(X[:,2] , X[:,3], c=labels.astype(np.int), alpha = 0.5)
plt.xlabel ('Anual Income (k$)')
plt.ylabel ('Spending Score (1-100)')
plt.show()


# # Hierarchical 

# In[301]:


print(df.dtypes)


# In[302]:


# Normalization
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(DF)
feature_mtx[0:5]


# In[303]:


# Applying Scikit Learn:
from sklearn.metrics.pairwise import euclidean_distances
dist_matrix = euclidean_distances(feature_mtx, feature_mtx)
print(dist_matrix)


# In[304]:


Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')


# In[305]:


fig = pylab.figure(figsize = (18,50))
def llf(id):
    return '[%s %s %s]' % (df['Age'][id], df['Annual Income (k$)'][id], int(float(df['Spending Score (1-100)'][id])) )
    
dendro = hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')


# In[306]:


Agglom = AgglomerativeClustering(n_clusters=6 , linkage='complete')
Agglom.fit(dist_matrix)
Agglom.labels_


# In[307]:


# To add the Clusters
df['Cluster (Hierarchical)'] = Agglom.labels_
df.head(10)


# # DBSCAN

# In[315]:


epsilon = 0.3
minimumSamples = 6
X1 = X[:, 2:3]
db = DBSCAN(eps=epsilon, min_samples=minimumSamples, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
DB = db.fit(X1)
labels = DB.labels_
labels


# In[316]:


# First, Create an array of booleans using the labels from DB
core_samples_mask = np.zeros_like(DB.labels_,dtype = bool)
core_samples_mask[DB.core_sample_indices_] = True
core_samples_mask


# In[317]:


# Number of Clusters in Labels, ignoring noise if present
print(set(labels))   # This is the total labels {0,1,2} are the cores, {-1} is the 
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_


# In[318]:


#Remove Repetition in labels by turning it into a set
unique_labels = set(labels)
unique_labels


# In[319]:


# Create Colors for Clusters
colors = plt.cm.Spectral(np.linspace(0,1,len(unique_labels)))


# In[325]:


# Plot the points with colors
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    # Plot the datapoints that are clustered
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:, 2], xy[:, 3],s=50, c=[col], marker=u'o', alpha=0.5)

    # Plot the outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:, 2], xy[:, 3],s=50, c=[col], marker=u'o', alpha=0.5)
    plt.xlabel ('Anual Income (k$)')
    plt.ylabel ('Spending Score (1-100)')


# ## It seems that DBSCAN Does not work as good as K-Mean and Hierarchical methods.
# ## I think K-means was the best option.

# In[ ]:




