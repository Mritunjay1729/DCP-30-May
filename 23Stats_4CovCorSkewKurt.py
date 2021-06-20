#Topic:Correlation, Covariance,
#-----------------------------
#libraries

#The difference between variance, covariance, and correlation is:

#Variance is a measure of variability from the mean
#Covariance is a measure of relationship between the variability of 2 variables - covariance is scale dependent because it is not standardized
#Correlation is a of relationship between the variability of of 2 variables - correlation is standardized making it not scale dependent


import pandas as pd
import numpy as np

import seaborn as sns

df = pd.DataFrame(np.random.randint(low= 0, high= 20, size= (5, 2)),  columns= ['Commercials Watched', 'Product Purchases'])
df

df.agg(["mean", "std", "max"], axis=1)

df

df.cov()

df.corr()

sns.heatmap(df.corr())

plt.scatter(df['Commercials Watched'], df['Product Purchases'])




#skewness & Kurtosis
import numpy as np
import pandas as pd
from scipy.stats import kurtosis


np.var(data)
np.mean(data)
np.std(data)

plt.hist(data, bins=60)

print("mean : ", np.mean(data))
print("var  : ", np.var(data))

print("skew : ",skew(data))

print("kurt : ",kurtosis(data))





import numpy as np
from scipy.stats import kurtosis, skew

x_random = np.random.normal(0, 2, 10000)
plt.hist(data, bins=60)

print("mean : ", np.mean(x_random))
print("var  : ", np.var(x_random))
print("skew : ",skew(x_random))
print("kurt : ",kurtosis(x_random))

plt.hist(x_random)



x_binom = np.random.binomial (size=10000,n=20,p=0.9)
plt.hist(x_binom, bins=60)

print("mean : ", np.mean(x_binom))
print("var  : ", np.var(x_binom))
print("skew : ",skew(x_binom))
print("kurt : ",kurtosis(x_binom))
plt.hist(x_binom)

x_binom = np.random.binomial (size=10000,n=20,p=1 )

print("mean : ", np.mean(x_binom))
print("var  : ", np.var(x_binom))
print("skew : ",skew(x_binom))
print("kurt : ",kurtosis(x_binom))


plt.hist(x_binom)




x_binom = np.random.binomial (size=10000,n=20,p=0.1 )

print("mean : ", np.mean(x_binom))
print("var  : ", np.var(x_binom))
print("skew : ",skew(x_binom))
print("kurt : ",kurtosis(x_binom))


plt.hist(x_binom)



x_binom = np.random.binomial (size=10000,n=20,p=0.5 )

plt.hist(x_binom, bins=60)

print("mean : ", np.mean(x_binom))
print("var  : ", np.var(x_binom))
print("skew : ",skew(x_binom))
print("kurt : ",kurtosis(x_binom))


plt.hist(x_binom)








x = np.linspace( -5, 5, 10000 )

#y = 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x)**2  ) 
 # normal distribution

import matplotlib.pyplot as plt

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(x_random, bins='auto')
ax1.set_title('probability density (random)')
ax2.hist(y, bins='auto')
ax2.set_title('(your dataset)')
plt.tight_layout()





#Basic statistics on MT Cars
import pandas as pd

import numpy as np

#read data
#df = pd.read_csv('data/mtcars.csv')
from pydataset import data

mtcars = data('mtcars')
mtcars.head()

df=mtcars
df.describe()
#df.dtypes()
#data distributions for 
df.columns

df.head(1)
#%%% =========================================
# #Skewness: It represents the shape of the distribution.
#Skewness can be quantified to define the extent to which a distribution differs from a normal distribution.
#For calculating skewness by using df.skew() python inbuilt function.

df.mpg
plt.hist(df.mpg, bins=10)

df.mpg.skew()  #positive : right skewed, moderate, right tail longer
df.mpg.kurtosis()
#majority of values in left of mean
#If skewness is not close to zero, then your data set is not normally distributed.If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.  If skewness is less than -1 or greater than 1, the distribution is highly skewed. If skewness is between -1 and -0.5 or between 0.5 and 1, the distribution is moderately skewed.If skewness is between -0.5 and 0.5, the distribution is approximately symmetric.

df.mpg.plot(kind='density')

#%%%
#Kurtosis: Kurtosis is the measure of thickness or heaviness of the given distribution.#Its actually represents the 
#height of the distribution.
#The distribution with kurtosis equal to 3 is known as mesokurtic. 
#A random variable which follows normal distribution has kurtosis 3.
#If the kurtosis is less than three, the distribution is called as platykurtic. 
#Here,the distribution has shorter and thinner tails than normal distribution.
#If the kurtosis is greater than three, the distribution is called as leptykurtic. 
#Here, the distribution has longer and fatter tails than normal distribution.
#For calculating kurtosis by using df.kurtosis() python inbuilt function.
# ========
#Baseline: Kurtosis value of 0

#Data that follow a normal distribution perfectly have a kurtosis value of 0. 
#Normally distributed data establishes the baseline for kurtosis. 
#Sample kurtosis that significantly deviates from 0 may indicate that the data are not normally distributed.


#meso kurtic : +3 between 
#https://www.quora.com/What-does-a-negative-kurtosis-indicates   : Read this

df.mpg.kurtosis()  # towards plateau away from normal
df.mpg.plot(kind='density')



#All columns
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 6))

df.columns

plt.subplot(2, 2, 1) # matrix of 2 x 2 plots : first plot
df["mpg"].plot.kde() 
df["mpg"].skew()
df["mpg"].kurtosis()

plt.title('Mileage')

plt.subplot(2, 2, 2) # matrix of 2 x 2 plots : 2nd plot
df.wt.plot.kde() 
df["wt"].skew()
df["wt"].kurtosis()


plt.title('Weight')
plt.subplot(2, 2, 3) # matrix of 2 x 2 plots : 3nd plot
df.hp.plot.kde()
df["hp"].skew()
df["hp"].kurtosis()
 
plt.title('Horse Power')
plt.subplot(2, 2, 4) # matrix of 2 x 2 plots : 4th plot
df["disp"].plot.kde() 
df["disp"].skew()
df["disp"].kurtosis()


plt.title('Displacement')
plt.show();

