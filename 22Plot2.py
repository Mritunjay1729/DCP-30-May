#Box Plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_1 = np.random.normal(100, 10, 200)

plt.plot(data_1)

data_1.min()
data_1.max()
data_1.mean()
data_1.std()

data_2 = np.random.normal(90, 20, 200)
data_2.mean()
 
data_3 = np.random.normal(80, 30, 200) 
data_3.mean()

data_4 = np.random.normal(70, 40, 200) 
data_4.mean()

data_5 = np.random.randint(50, 100, 10)
data_5.mean()
data_5.sort()
data_5


data_7 = np.random.normal(70, 10, 11) 
data_7.mean()
data_7.sort()
data_7

data_6 = np.array([1,2,3,8,9])
data_6.mean()
data_6.sort()
data_6




data = [data_1, data_2, data_3, data_4, data_5] 

data  =[data_1]

fig = plt.figure(figsize =(5, 5)) 
# Creating axes instance 
ax = fig.add_axes([1,2,3,4]) 
ax

# Creating plot 
bp = ax.boxplot(data)

# show plot 
plt.show() 




l1 = [1,4,7,9,10]
np.mean(l1)


#Different dataset
import pandas as pd
data = pd.read_csv('brain_size.csv', sep=';', na_values='.')
data.head()
# Box plot of FSIQ and PIQ (different measures od IQ)
plt.figure(figsize=(4, 3))
data.columns

data.boxplot(['FSIQ', 'PIQ', 'VIQ'])

plt.show();

data
data['FSIQ'] - data['PIQ']
# Boxplot of the difference
plt.figure(figsize=(4, 3))
plt.boxplot(data['FSIQ'] - data['PIQ'])

plt.xticks((1, 0, 2), ('FSIQ - PIQ','VV', 'AA' ))

plt.yticks((-10, -15,), ("BB", "AA",))

plt.show();


