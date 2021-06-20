# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 20:16:55 2021

@author: vikas
"""

import matplotlib.pyplot as plt

Year = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010]
Unemployment_Rate = [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]
Year
Unemployment_Rate

plt.plot(Year, Unemployment_Rate)
plt.title('Year vs Unemployment_Rate', fontsize=14)
plt.xlabel('Year')
plt.ylabel('Unemployment_Rate')
plt.grid(True)
plt.legend()
plt.show()


import pandas as pd
Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010], 'Unemployment_Rate': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3]}
Data  
df = pd.DataFrame(Data,columns=['Year','Unemployment_Rate'])
df  

plt.plot(df['Year'], df['Unemployment_Rate'], color='red', marker='*')
plt.title('Unemployment Rate Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate', fontsize=14)
plt.grid(True)
plt.show()





import pandas as pd
Data = {'Year': [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010], 'UR': [9.8,12,8,7.2,6.9,7,6.5,6.2,5.5,6.3],
        'UR1': [1.8,14,8,9.2,4.9,9,5.5,8.2,2.5,6.3]
        }
Data  
df = pd.DataFrame(Data,columns=['Year','UR','UR1' ])
df  

plt.plot(df['Year'], df['UR'], color='red', marker='*')
plt.plot(df['Year'], df['UR1'], color='blue', marker='o')

plt.title('Unemployment Rate Vs Year', fontsize=14)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Unemployment Rate', fontsize=14)
plt.grid(True)
plt.legend(['UR1','UR2'])
plt.show()


Country = ['USA','Canada','Germany','UK','France']
GDP_Per_Capita = [45000,42000,52000,49000,47000]

plt.bar(Country, GDP_Per_Capita)
plt.title('Country Vs GDP Per Capita')
plt.xlabel('Country')
plt.ylabel('GDP Per Capita')
plt.grid(True)

plt.show();


New_Colors = ['green','blue','purple','brown','teal']

plt.bar(Country, GDP_Per_Capita, color=New_Colors)
plt.title('Country Vs GDP Per Capita', fontsize=14)
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Per Capita', fontsize=14)
plt.grid(True)
plt.show();


Data = {'Country': ['USA','Canada','Germany','UK','France'], 'GDP_Per_Capita': [45000,42000,52000,49000,47000]    }
df = pd.DataFrame(Data,columns=['Country','GDP_Per_Capita'])
df

New_Colors = ['green','blue','purple','brown','teal']
plt.bar(df['Country'], df['GDP_Per_Capita'], color=New_Colors)
plt.title('Country Vs GDP Per Capita', fontsize=14)
plt.xlabel('Country', fontsize=14)
plt.ylabel('GDP Per Capita', fontsize=14)
plt.grid(True)
plt.show();




'''    
fig, ax = plt.subplots(nrows=1, ncols=1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index

ax = plt.gca()
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((1.0, 0.47, 0.42))
'''


import numpy as np
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

x
y

# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True)
f
ax1
ax1.plot(x, y)

ax2.scatter(x, y)


f, ax = plt.subplots(2, 2, sharey=True, sharex=True)
ax

ax[0,0].plot(x,y)
ax[1,0].scatter(x,y)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt?

#%%basic scatter plot
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x,y)

from pydataset import data
data()
mtcars = data('mtcars')
#conda upgrade --all -y
df=mtcars

df.describe

df.dtypes
df.columns
df['mpg']
df['wt']


plt.scatter(df.wt, df.mpg)
plt.scatter(df.hp, df.mpg)
























