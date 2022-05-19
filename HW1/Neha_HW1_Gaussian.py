#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In the section below, we are importing libraries which will be used in the program below
# <numpy> to compute mean error for the predicted values
# <pandas> to load and parse the csv file into meaningful dataframes
import numpy as np
import pandas as pd


# In[2]:


# In the section below, we are loading training data csv file into dataframe named df_train.
df_train = pd.read_csv('zip.train.p.csv')
print(df_train.head())


# In[3]:


# In the section below, we are separating attributes and labels for training data.  
X_train = df_train.iloc[:, 1:256].values
Y_train = df_train.iloc[:, 0].values


# In[4]:


# In the section below, we are loading test data csv file into dataframe named df_test.
df_test = pd.read_csv('zip.test.p.csv')
df_test.head()


# In[5]:


# In the section below, we are separating attributes and labels for test data.
X_test = df_test.iloc[:, 1:256].values
Y_test = df_test.iloc[:, 0].values


# In[6]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Create a Gaussian Classifier
model = GaussianNB()

#Train the model using the training sets
model.fit(X_train,Y_train)

#Predict Output
Y_pred= model.predict(X_test)


# In[7]:


# Print out classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(Y_test, Y_pred))
# Print out confusion matrix
print(confusion_matrix(Y_test, Y_pred))

