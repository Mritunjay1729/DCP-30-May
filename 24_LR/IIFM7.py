# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:58:46 2020

@author: vikas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("house.csv")

df

df = df.dropna()
df

plt.scatter(df['area'], df['Price'])


from sklearn.linear_model import LinearRegression

model = LinearRegression()

x= df['area'].values
y= df['Price'].values

x.shape

x=x.reshape(-1,1)
x.shape

model.fit (x, y)

y_pred = model.predict(x)

y_pred

plt.scatter(x, y)
plt.scatter(x, y_pred)


x_test = np.array([2500, 2700, 2950, 3300]).reshape(-1,1)

x_test.shape

y_pred1 = model.predict(x_test)

y_pred1


plt.scatter(x, y)
plt.scatter(x, y_pred)
plt.scatter(x_test, y_pred1)


r_sq = model.score(x, y)

r_sq





#Example 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

mt = data('mtcars')

mt.head()
mt.columns

x = mt.hp.values.reshape(-1,1)
y = mt.disp.values

x.shape
y.shape

plt.scatter(x,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(x,y)
plt.scatter(x,y_pred)


r_sq = model.score(x,y)

r_sq




#Example 3 Multi LEanear Regression

mt.head()

x = mt[['mpg', 'hp']].values

x.shape

y = mt['disp'].values

y.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x)

y_pred

r_sq = model.score(x,y)
r_sq



from  sklearn.model_selection import train_test_split

x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3)

x_train.shape
x_test.shape
y_train.shape
y_test.shape



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred

r_sq = model.score(x_test,y_test)
r_sq





#Example 4


df = pd.read_csv("hourly_wages.csv")
df.columns


y= df.wage_per_hour.values

x = df[['union', 'education_yrs', 'experience_yrs', 'age',
       'female', 'marr', 'south', 'manufacturing', 'construction']].values

x.shape

y.shape


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1)

x_train.shape
x_test.shape
y_train.shape
y_test.shape



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred

r_sq = model.score(x_test,y_test)
r_sq




#Example

df = pd.read_csv(("insure.csv"))

df.columns


x= df[['age', 'sex', 'bmi', 'children', 'smoker']].values

x

y = df['charges'].values



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.1)

x_train.shape
x_test.shape
y_train.shape
y_test.shape



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred

r_sq = model.score(x_test,y_test)
r_sq



from sklearn.metrics import r2_score,mean_squared_error

r2_score(y_test, y_pred)

mean_squared_error(y_test, y_pred)






#Example


# -*- coding: utf-8 -*-

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc


df = pd.read_csv("Social_Network_Ads.csv")

df.columns

df = df.drop(['User ID', 'Gender', 'Age'], axis=1)


y = df['Purchased']

y


df = df.drop(['Purchased'], axis =1)

X = df

df.dtypes

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
y_pred.shape
y_test.shape

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




y_pred_proba = logreg.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();





#Example


import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.metrics import roc_curve, auc


df = pd.read_csv("titanic_all_numeric.csv")

df.columns


y = df['survived']

y


df = df.drop(['survived'], axis =1)

X = df

df.dtypes

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)
y_pred.shape
y_test.shape

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))




y_pred_proba = logreg.predict_proba(X_test)[::,1]
y_pred_proba
y_pred_proba.shape


fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)

fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

#together

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show();















# RandomForest and Descision Tree Regressors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

mt = data('mtcars')

mt.head()

x = mt[['mpg', 'hp']].values

x.shape

y = mt['disp'].values

y.shape


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

model.fit(x, y)

y_pred = model.predict(x)

y_pred

r_sq = model.score(x,y)
r_sq



from  sklearn.model_selection import train_test_split

x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3)

x_train.shape
x_test.shape
y_train.shape
y_test.shape



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred

r_sq = model.score(x_test,y_test)
r_sq






#Linear Regression Comparison with RandomForest and Descision Tree Regressors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

mt = data('mtcars')

mt.head()

x = mt[['mpg', 'hp']].values

x.shape

y = mt['disp'].values

y.shape


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x)

y_pred

r_sq = model.score(x,y)
r_sq



from  sklearn.model_selection import train_test_split

x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3)

x_train.shape
x_test.shape
y_train.shape
y_test.shape



from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred

r_sq = model.score(x_test,y_test)
r_sq



# randomforestRegressor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data

mt = data('mtcars')

mt.head()

x = mt[['mpg', 'hp']].values

x.shape

y = mt['disp'].values

y.shape



from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(x, y)

y_pred = model.predict(x)

y_pred

r_sq = model.score(x,y)
r_sq












































































































