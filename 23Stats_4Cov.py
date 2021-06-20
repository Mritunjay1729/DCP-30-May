#Topic: Statistics - Covariance
#-----------------------------
#Covariance provides the a measure of strength of 
#correlation between two variable or more set of variables.
# The covariance matrix element Cij is the covariance of xi and xj. 
#The element Cii is the variance of xi.

#If COV(xi, xj) = 0 then variables are uncorrelated
#If COV(xi, xj) > 0 then variables positively correlated
#If COV(xi, xj) < 0 then variables negatively correlated

#numpy.cov(m, y=None, rowvar = True, bias=False, ddof=None, fweights=None, aweights=None)


#Parameters:
#m : [array_like] A 1D or 2D variables. variables are columns
#y : [array_like] It has the same form as that of m.
#rowvar : [bool, optional] If rowvar is True (default), then each row represents a variable, with observations in the columns. Otherwise, the relationship is transposed:
#bias : Default normalization is False. If bias is True it normalize the data points.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

A = [45,37,42,35,39]
B = [38,31,26,28,33]
C = [10,15,17,21,12]

data = np.array([A,B,C])

data


covMatrix = np.cov(data)
print (covMatrix)


sns.heatmap(covMatrix, annot=True, fmt='f')
plt.show()