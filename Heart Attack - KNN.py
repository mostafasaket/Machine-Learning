#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


# In[75]:


df = pd.read_csv('heart.csv')
df


# In[76]:


df.dtypes


# In[77]:


X_pd = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']]
X = np.asarray(X_pd)
print(X[0:2] , 'Shape of X: ', X.shape)

Y_pd = df[['output']]
Y = np.asarray(Y_pd)
print(Y[0:2], 'Shape of Y: ', Y.shape)


# In[78]:


scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X.astype(int))
print ('X_Normalized:' , X[0:2])


# In[79]:


from sklearn.model_selection import train_test_split
X_train, X_test , Y_train, Y_test = train_test_split (X,Y, test_size = 0.2 , random_state = 4)
print ('Train Set:' , X_train.shape , Y_train.shape)
print ('Train Set: ', X_test.shape , Y_test.shape) 


# In[80]:


from sklearn.neighbors import KNeighborsClassifier
k = 5
KNN = KNeighborsClassifier(weights= 'distance', n_neighbors= k)
NEIGH = KNN.fit(X_train , Y_train)
NEIGH


# In[81]:


Y_hat = NEIGH.predict(X_test)
print('Predicted Values: ' , Y_hat[0:5])
print('Actual Values: ' , Y_test[0:5].T)


# In[82]:


from sklearn import metrics
print('Train Set Accuracy: ', metrics.accuracy_score(Y_train,NEIGH.predict(X_train)))  # Accuracy based on the Trains Set
print('Test Set Accuracy: ', metrics.accuracy_score(Y_test, Y_hat))  # Accurcy based on the Test Set


# In[ ]:





# In[83]:


from sklearn.metrics import classification_report , confusion_matrix
import itertools 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[72]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, Y_hat, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(Y_test, Y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Low Risk','High Risk'],normalize= False,  title='Confusion matrix')


# In[ ]:




