# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 19:03:54 2021

@author: vikas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

from sklearn.linear_model import LinearRegression

mtcars = data('mtcars')


data=mtcars
data.head()
data.columns
data.dtypes
data.shape


ds= mtcars[['mpg', 'disp']]

ds.head()

plt.scatter(ds.mpg, ds.disp)

x = ds.mpg.values.reshape((-1,1))

y = ds.disp.values

model = LinearRegression()

model.fit(x,y)

model.score(x,y)

x_pred = np.array([19.8, 50, 73]).reshape((-1,1))
x_pred.shape


y_pred  = model.predict(x_pred)

plt.scatter(ds.mpg, ds.disp)
plt.scatter(x_pred, y_pred)


import seaborn as sns
sns.distplot(x)
sns.distplot(y)


data.dtypes

ds= mtcars[['mpg', 'drat']]

ds.head()

plt.scatter(ds.mpg, ds.drat, ds.disp)



ds= mtcars[['mpg', 'drat', 'disp']]

x = ds[['drat', 'disp']].values

y = ds.mpg.values

x


model = LinearRegression()

model.fit(x,y)

model.score(x,y)



x_pred = np.array([[3,415],[5,560]])

y_pred = model.predict(x_pred)
y_pred




ds= mtcars[['mpg', 'drat', 'qsec']]

x = ds[['drat', 'qsec']].values

y = ds.mpg.values




model = LinearRegression()

model.fit(x,y)

model.score(x,y)




import statsmodels.api as sm
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
from pydataset import data



mtcars = data('mtcars')
mtcars.columns
df1 = mtcars[['wt','mpg']]
df1.head(5)


from statsmodels.formula.api import ols


MTmodel1 = ols("mpg ~ wt).fit()
print(MTmodel1.summary())


pred = MTmodel1.predict()

pred


plt.scatter(df1.mpg, df1.wt)
plt.scatter(pred, df1.wt)


x_pred = np.array([40, 42, 54])


y_pred= MTmodel1.predict(x_pred)






import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm

plt.rc("figure", figsize=(16,8))
plt.rc("font", size=14)


y = df1.mpg.values

X = df1.wt.values

X.shape

x_pred.shape

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
print(olsres.summary())

ypred = olsres.predict(x_pred)
print(ypred)


plt.scatter(X,y)
plt.scatter(x_pred, ypred)






import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
# Load the diabetes dataset

'''
url = "https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/slr1.csv"
data = pd.read_csv(url)
#data = pd.read_csv('data/slr1.csv')
data.columns
data.head(1)
'''

data = pd.read_csv('age.csv')
#data = pd.read_csv('data/slr1.csv')
data.columns
data.head(1)

x = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
       'DiabetesPedigreeFunction']].values

x.shape

y = data['Age'].values

y.shape


import statsmodels.api as sm
olsmod = sm.OLS(y, x)
olsres = olsmod.fit()
print(olsres.summary())

x[1]

x_pred =  np.array([[80, 61, 35, 1 , 26.6 , 0.551],[80, 61, 35, 1 , 26.6 , 0.551]])

y_pred = olsres.predict(x_pred)
y_pred


from sklearn.model_selection import train_test_split

x_train, x_test, y_train,  y_test = train_test_split(x,y, test_size= 0.1)

x_train.shape, x_test.shape, y_train.shape,  y_test.shape


import statsmodels.api as sm
olsmod = sm.OLS(y_train, x_train)
olsres = olsmod.fit()
print(olsres.summary())


y_pred = olsres.predict(x_test)

y_pred

y_test



model = LinearRegression()

model.fit(x_train, y_train)

y_pred  = model.predict(x_test)

y_pred









#Topic: Linear Regression Stock Market Prediction 
#-----------------------------
#libraries
import pandas as pd
import matplotlib.pyplot as plt

Stock_Market = {'Year': [2017,2017,2017, 2017,2017,2017,2017,2017, 2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016, 2016,2016,2016], 'Month': [12, 11,10,9,8,7,6, 5,4,3, 2,1,12,11, 10,9,8,7,6,5,4,3,2,1], 'Interest_Rate': [2.75,2.5,2.5,2.5,2.5, 2.5,2.5,2.25,2.25, 2.25,2,2,2,1.75,1.75, 1.75,1.75, 1.75,1.75,1.75,1.75,1.75,1.75,1.75], 'Unemployment_Rate': [5.3,5.3, 5.3,5.3,5.4,5.6,5.5, 5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1, 5.9,6.2,6.2, 6.1],'Stock_Index_Price':[1464,1394,1357,1293,1256,1254,1234,1195, 1159,1167,1130,1075,1047,965, 943,958,971,949,884,866,876,822,704,719]   }  #dictionary format
type(Stock_Market)


df = pd.DataFrame(Stock_Market, columns=['Year','Month','Interest_Rate', 'Unemployment_Rate','Stock_Index_Price']) 
df.head()
print (df)

#check that a linear relationship exists between the:
#Stock_Index_Price (dependent variable) and Interest_Rate (independent variable)
#Stock_Index_Price (dependent variable) and Unemployment_Rate (independent variable)

#run these lines together
plt.scatter(df['Interest_Rate'], df['Stock_Index_Price'], color='red')
plt.title('Stock Index Price Vs Interest Rate', fontsize=14)
plt.xlabel('Interest Rate', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show();
# linear relationship exists between the Stock_Index_Price and the Interest_Rate. Specifically, when interest rates go up, the stock index price also goes up:
    
plt.scatter(df['Unemployment_Rate'], df['Stock_Index_Price'], color='green')
plt.title('Stock Index Price Vs Unemployment Rate', fontsize=14)
plt.xlabel('Unemployment Rate', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
plt.show() ;   


#linear relationship also exists between the Stock_Index_Price and the Unemployment_Rate – when the unemployment rates go up, the stock index price goes down (here we still have a linear relationship, but with a negative slope):

#Multiple Linear Regression
from sklearn import linear_model #1st method
    
X = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example. Alternatively, you may add additional variables within the brackets
Y = df['Stock_Index_Price']

# with sklearn
regr = linear_model.LinearRegression()

regr.fit(X, Y)


y_pred= regr.predict(X.values)

y_pred
Y

from sklearn.metrics import  r2_score

r2_score(Y, y_pred)


df


New_Interest_Rate = 4.75
New_Unemployment_Rate = 3.3

print ('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate ,New_Unemployment_Rate]]))

data1=X
data1['R']=Y
data1.columns



from statsmodels.formula.api import ols

model2 = ols('R ~ Interest_Rate + Unemployment_Rate', data=data1).fit()

model2.summary()






df = pd.read_csv('data1.csv')

df


plt.scatter(df.x, df.y)

x =df.x.values.reshape((-1,1))

y = df.y.values
model = LinearRegression()


model.fit(x, y)

y_p = model.predict(x)
y_p

plt.scatter(x, y)
plt.scatter(x, y_p)





















