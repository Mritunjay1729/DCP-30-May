# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:00:24 2021

@author: vikas
"""

import pandas as pd # used to create the DataFrame to capture the dataset in Python
#sklearn    # used to build the logistic regression model in Python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
import seaborn as sns # used to create the Confusion Matrix
import matplotlib.pyplot as plt # used to display charts

#data from csv
url = "https://stats.idre.ucla.edu/stat/data/binary.csv"
df = pd.read_csv(url)

df
df.drop_duplicates()

X = df[['gpa', 'gre']].values

y = df['admit'].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
X_train.shape, X_test.shape
y_train.shape, y_test.shape





model= LogisticRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)




from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(y_test, y_pred)

cf =confusion_matrix(y_test, y_pred)

(53+2)/(53+5+2+20)


sns.heatmap(cf, annot=True, cmap='coolwarm', annot_kws={'size':20}, cbar=True)
plt.show();





'''

def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))

res = []

for i in X_train:
    res.append(sigmoid(i))

res


plt.scatter(X_train,y_train)
plt.scatter(X_train,res)

'''




y_pred=logistic_regression.predict(X_test)

y_pred

















