# -*- coding: utf-8 -*-
#Heat Map
#A heatmap contains values representing various shades of the same colour for each value to be plotted. Usually the darker shades of the chart represent higher values than the lighter shade. For a very different value a completely different colour can also be used.
#The below example is a two-dimensional plot of values which are mapped to the indices and columns of the chart.

from pandas import DataFrame
import matplotlib.pyplot as plt
#data for 4 columns
data=[{2,3,4,1},{6,3,5,2},{6,3,5,4},{3,7,5,4},{2,8,1,5}]

Index= ['I1', 'I2','I3','I4','I5']  #Index values
Cols = ['C1', 'C2', 'C3','C4']  #columns
df = DataFrame(data, index=Index, columns=Cols)
df


plt.pcolor(df)
plt.show()

import seaborn as sns
sns.heatmap(df)



#1-darkest, 8 - lightest
    
#https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

#https://seaborn.pydata.org/generated/seaborn.heatmap.html
import numpy as np
import seaborn as sns
import pandas as pd
uniform_data = np.random.rand(10, 10)
uniform_data

df = pd.DataFrame(uniform_data)

ax = sns.heatmap(uniform_data)

ax= sns.heatmap(df)
df

fig, axis = plt.subplots() # il me semble que c'est une bonne habitude de faire supbplots
heatmap = axis.pcolor(uniform_data, cmap=plt.cm.Blues) # heatmap contient les valeurs
plt.colorbar(heatmap)


#Histogram - Matplotlib
##-----------------------------
#%
#mtcars data

from pydataset import data

mtcars = data('mtcars')

mtcars.head(5)
mtcars
mtcars.dtypes

#which are continuous variables
mtcars[['mpg','wt','hp','disp']]. head()

mtcars['mpg']

import matplotlib.pyplot as plt
import numpy as np

plt.hist(mtcars.mpg)
plt.hist(mtcars['mpg'])

#other options
plt.hist(mtcars['mpg'], alpha=0.1)

plt.hist(mtcars.wt)


mtcars.wt.mean()
mtcars.wt.max()
mtcars.wt.min()

np.linspace(mtcars.wt.min(),mtcars.wt.max(), num=10)

np.histogram(mtcars.wt, bins=10)

plt.hist(mtcars.wt)


import seaborn as sns

sns.distplot(mtcars['mpg'], kde=True)



data_2 = np.random.normal(90, 20, 200)
data_2

sns.distplot(data_2, kde=True)

