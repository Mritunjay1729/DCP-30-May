# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 22:08:31 2021

@author: vikas
"""

import pandas as pd

df = pd.read_csv('insure.csv')

df.columns


x = df[['age', 'sex', 'bmi', 'children', 'smoker']].values

y=df['charges'].values

x.shape
y.shape


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

x_train.shape
y_train.shape
x_test.shape
y_test.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train,y_train)

r2 = model.score(x_train,y_train)
r2

r2 = model.score(x_test,y_test)
r2

y_pred = model.predict(x_test)


import seaborn as sns
# Pair-wise Scatter Plots
df = df[['age', 'sex', 'bmi', 'children', 'smoker','charges']]

cols = list(df.columns)

pp = sns.pairplot(df[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))





import matplotlib.pyplot as plt

plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)
plt.scatter(x_test, y_pred)




