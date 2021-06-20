# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 23:04:18 2021

@author: vikas
"""

import math

math.exp(10)

ls= list(range(-16,16))

y=[]
for x in ls:
    y.append((1)/(1+math.exp(-x)))




import matplotlib.pyplot as plt

plt.scatter(ls,y)
