# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


df = pd.read_csv("dat.csv")

df

X= df.X.values.reshape(-1,1)
Y = df.Y.values


X_train,  X_test, Y_train, Y_test = train_test_split(X,Y,test_size =0.3)

X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

model = LinearRegression()

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

y_pred
Y_test


plt.scatter(X_test, Y_test )
plt.scatter(X_test, y_pred )


model.score(X,Y)



