# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 21:13:50 2021

@author: vikas
"""

[1,2,3,4,5,6]


import random as rd

r1 = rd.randint(1,10000)
print(r1)


l1 = [111,112,444,555,999,333,556]

print(rd.choice(l1))

l2 = list(range(1,10000000))

print(rd.choices(l2, k=10000))

l3 = ['Amritsar', 'Jalandhar', 'Mumbai', 'chandigarh', 'Cochin']

rd.choices(l3, k=3)


rno = list(range(1,101))
gender = ['M','F']


r = rd.choices(rno, k=1000)
r

g = rd.choices(gender, k=1000)
g

d = {'rno':r, 'gen':g}
d

import pandas as pd
pd.DataFrame(d)

















