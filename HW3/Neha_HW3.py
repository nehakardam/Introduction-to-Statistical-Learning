#!/usr/bin/env python
# coding: utf-8

# #### Neha_HW 3_Part II: Computer Assignment Solution
# 
# I used CountVectorizer to represent the documents as bag of words with occurance counting. I found 73686 words in the training data and ruled out the ones which occured only a few times (<= 10). I left with 15226 words for which I computed mutual information and selected the top 5000 words. I represented my dataset using log-normalized counts where each entry becomes log(td + 1). I used LinearSVC classifier which handles multiclass support using one-vs-the-rest scheme. I did a cross-validation for 2 classifiers and found the one with L2 penality performing better. I then evaluate by best classifier using test data and found 74% accuracy. I checked five largest outliers in the confusion matrix and noticed that our classifier is predicting wrong class where they are similar in nature (for ex. predicting talk.politics.guns instead of talk.politics.misc). I also checked top 10 and bottom 10 features for each class, and cross-checked them with our mutual information based selected features.

# In[1]:


import numpy as np
import pandas as pd
import math


# #### (a) Setup - loading training data

# In[2]:


df = pd.read_csv('20ng-train-all-terms.csv', names=['class_publication_name', 'document'])
df.head()


# #### (b) Vocabulary Selection - coverting the given document text into features (word) using CountVectorizer

# In[3]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df.document.to_numpy())
word_count_df = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names())
word_count_df.head()


# #### (b) Vocabulary Selection - ruling out words that occur only a few times (i.e. word_count <= 10 in the entire corpus)

# In[4]:


frequent_word_df = pd.DataFrame()
for i in range(0, word_count_df.shape[1]):
    if(word_count_df.iloc[:,i].sum() > 10):
        frequent_word_df[word_count_df.iloc[:,i].name] = word_count_df.iloc[:,i]
frequent_word_df.head()


# #### (b) Vocabulary Selection - selecting top 5000 words by mutual information

# In[5]:


from sklearn.feature_selection import mutual_info_classif

X_train = frequent_word_df
Y_train = df.class_publication_name
top_features = []
feature_to_mi = {}

mutual_info = mutual_info_classif(X_train, Y_train, discrete_features=True)

for i in range(0,len(mutual_info)):
    feature_to_mi[frequent_word_df.columns.values[i]] = mutual_info[i]
i = 0;
for k in sorted(feature_to_mi, key=feature_to_mi.get, reverse=True):
    if(i==5000):
        break
    else:
        top_features.append(k)
        i = i+1


# #### (b) Vocabulary Selection - table of the top ten words by their mutual information

# In[6]:


print("Top 10\t", "Mutual Information")
for i in range (0, 10):
    print(top_features[i],"\t", feature_to_mi[top_features[i]])


# #### (c) Input Representation - log-normalized counts where each entry becomes log(td + 1).

# In[7]:


X_train_best = pd.DataFrame()
for feature in top_features:
    X_train_best[feature] = X_train[feature]
X_train_best = np.log(X_train_best + 1)
X_train_best.head()


# #### (d) Classifier - using linear support vector machine classifiers and using 5-Fold cross validation

# In[8]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

#clf1 = LinearSVC(penalty="l1", dual=False, tol=1e-3)
clf2 = LinearSVC(penalty="l2", dual=False, tol=1e-3)

#print('clf1', cross_val_score(clf1, X_train_best, Y_train, n_jobs=-1))
print('clf2', cross_val_score(clf2, X_train_best, Y_train, n_jobs=-1))


# #### From the cross validation above, we can see classifier 2 (clf2) is better

# In[9]:


clf2.fit(X_train_best, Y_train)


# #### (e) Evaluation - load and process the test data

# In[15]:


df_test = pd.read_csv('20ng-test-all-terms.csv', names=['class_publication_name', 'document'])
X_test_best = pd.DataFrame()
X_test = vectorizer.fit_transform(df_test.document.to_numpy())
word_count_df_test = pd.DataFrame(X_test.toarray(), columns = vectorizer.get_feature_names())
for feature in top_features:
    if feature in word_count_df_test:
        X_test_best[feature] = word_count_df_test[feature]
    else:
        X_test_best[feature] = 1
X_test_best = np.log(X_test_best + 1)
X_test_best.head()


# #### (e) Evaluation - evaluate using test data and report the accuracy

# In[16]:


Y_test = df_test.class_publication_name
Y_pred = clf2.predict(X_test_best)
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# #### (e) Evaluation - five largest off-diagonal entries

# In[25]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred)
top_5 = [0,0,0,0,0]
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[1]):
        if((i != j) & (confusion_matrix[i][j] > top_5[0])):
            top_5.pop(0)
            top_5.append(confusion_matrix[i][j])
            top_5.sort()
print(confusion_matrix,'\n', top_5)


# Five largest off-diagonal entries are:
# 1) 87 times talk.politics.misc got predicted as talk.politics.guns
# 2) 42 times talk.religion.misc got predicted as soc.religion.christians
# 3) 37 times comp.windows.x got predicted as comp.graphics
# 4) 36 times comp.os.ms-windows.misc got predicted as comp.graphics
# 5) 35 times comp.sys.ibm.pc.hardware got predicted as comp.sys.mac.hardware 
# 
# All of the above examples shows that our model sometimes predicts similar clases from the to same domain (for ex. predicting talk.politics.guns instead of talk.politics.misc).

# #### (f) Model Inspection - table of top 10 and bottom 10 features for each class

# In[18]:


frequent_word_df_expanded = frequent_word_df.copy()
frequent_word_df_expanded.insert(0, 'class_publication_name', df.class_publication_name)
frequent_word_df_weighted = frequent_word_df_expanded.groupby('class_publication_name').sum()
frequent_word_df_weighted.head()


# In[30]:


for i in range(0,20):
    top_10 = [0,0,0,0,0,0,0,0,0,0]
    top_10_features = []
    bottom_10 = [100,100,100,100,100,100,100,100,100,100]
    bottom_10_features = []
    for j in range (0, frequent_word_df_weighted.shape[1]):
        if(frequent_word_df_weighted.iloc[i][j] > top_10[0]):
            top_10.pop(0)
            top_10.append(frequent_word_df_weighted.iloc[i][j])
            top_10_features.append(frequent_word_df_weighted.columns.values[j])
            top_5.sort()
        if(frequent_word_df_weighted.iloc[i][j] < bottom_10[9]):
            bottom_10.insert(0,frequent_word_df_weighted.iloc[i][j])
            bottom_10_features.insert(0, frequent_word_df_weighted.columns.values[j])
            bottom_10.sort()
    print(frequent_word_df_weighted.index.values[i], "top_10", top_10_features[-10:])
    print(np.in1d(top_10_features[-10:], top_features))
    print(frequent_word_df_weighted.index.values[i], "bottom_10", bottom_10_features[:10])
    print(np.in1d(bottom_10_features[:10], top_features))


# The top 10 and bottom 10 features against each class are inlined with what we found using mutual information. All the top 10 features are in our list of top 5000 workds as per mutual information where as none of the bottom 10 features are there in the list.
