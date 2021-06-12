# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 21:42:41 2021

@author: vikas
"""

import numpy as np


np.random.randint(100, 1000)

x1 = np.random.randint(100, 200, size=10)
x1

x1.shape

x2= np.random.randint(100, 200, size= (3,5))
x2
x2.shape


x1
x1[:]
x1[2:]
x1[:5]
x1[:-2]
x1
x1[2:-2]
x1[-5:-2]
x1[6:-2]

x2= np.random.randint(100, 200, size= (3,5))
x2
x2.shape


x2[0][0]

x2[2][4]

x2[2,4]

x2
x2[0:2,0]
x2[0:2,0:2]
x2[1:3,1:3]



sum(x2[1])

i=0
while(i<=2):
    print(sum(x2[i]))
    i = i+1


sum(x2[0:2,0:2])

x2[-2,-2:]


x3= np.random.randint(100, 200, size= (1,3,4))
x3
x3.shape


x3[1,0,0]

x3[0][0][0]

min(x3[1,0:2,1])


x2
min(x2[:,0])


x2
x2.shape

x2.reshape((5,3))




x2= np.random.randint(100, 200, size= (3,4))
x2
x2.shape

x2.reshape(6,2)

x2.reshape(4,3)

x2.reshape(2,6)



a = np.zeros((2,4))
a

b = np.zeros((2,4), dtype='int')
b

c = np.ones((2,3), dtype='int')
c


n1 = np.empty((2,4))
n1


d = np.eye(3,3)
d


d[1,1]=30
d

l1 =np.linspace(-10,10,num=5)
l1


'''
x1 = np.random.normal(loc=0, scale=1, size=1000000 )
len(x1)

import matplotlib.pyplot as plt

plt.hist(x1)
'''



