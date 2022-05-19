#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In the section below we are importing libraries which will be used in the program below
# <numpy> to compute mean error for the predicted values
# <matplotlib.pyplot> to plot the error graph
# <pandas> to load and parse the csv file into meaningful dataframes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# In the section below we are loading training data csv file into dataframe named df_train.
df_train = pd.read_csv('zip.train.p.csv')
df_train.head()


# In[3]:


# In the section below we are separating attributes and labels for training data.  
X_train = df_train.iloc[:, 1:256].values
Y_train = df_train.iloc[:, 0].values


# In[4]:


# In the section below we will use 10% of the training data as validation data.
from sklearn.model_selection import train_test_split
# split dataset into training and validation data
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=1, stratify=Y_train)


# In[17]:


# In the section below we are training and validating KNN classifier ...
# ... for different values of K (1 to 20).
from sklearn.neighbors import KNeighborsClassifier
error = []
min_error = 1
best_knn = KNeighborsClassifier(n_neighbors=0)
# find the best classifier for k = {1:20} based on minimum mean error.
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    # training the model
    knn.fit(X_train, Y_train)
    # predicting the label using test data
    y_pred = knn.predict(X_validation)
    error_i = np.mean(y_pred != Y_validation)
    error.append(error_i)
    if(error_i < min_error):
        print(i, error_i)
        min_error = error_i
        best_knn = knn


# In[18]:


# In the section below we are plotting to visualize mean error agaist different values of K.
plt.figure(figsize=(12, 6))
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[19]:


# In the section below we are loading test data csv file into dataframe named df_test.
df_test = pd.read_csv('zip.test.p.csv')
print(df_test.head())


# In[20]:


# In the section below we are separating attributes and labels for test data.
X_test = df_test.iloc[:, 1:256].values
Y_test = df_test.iloc[:, 0].values


# In[21]:


# let's predict the labels using the best model
Y_pred = best_knn.predict(X_test)


# In[22]:


# Print out classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test, Y_pred))
# Print out confusion matrix
print(confusion_matrix(Y_test, Y_pred))

