# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:01:18 2021

@author: vikas
"""

import pandas as pd


#import pydataset

from pydataset import data

data('')

mt = data('mtcars')
mt

type(mt)

mt.head()
mt.tail()

mt.head(10)
mt.tail(10)


mt
mt.to_csv('mtcars.csv')
mt.to_excel('mtcars.xlsx')




import pandas as pd

df = pd.read_csv('mtcars.csv')

df

df1 = pd.read_excel('mtcars.xlsx')
df1

df1.shape

df1.columns


r = range(1,11)

s = pd.Series(r)
s

type(s)

r = range(101,133)

s = pd.Series(r)
s
mt

mt.set_index(s)


ps1 = pd.Series([1,4,8,10,33], dtype='float')
ps1


ps1 = pd.Series([1,4,8.6,10.7,33], dtype='float')
ps1

ps1[0]

ps1[1]

ps1[1]=20
ps1


ps1[1:4]


ps2 = pd.Series([1,4,8,11,33])
ps2

ps2 = pd.Series([1,4,8,11,33], index=['a','b','c','d','e'])
ps2

ps2['a']

ps3 = pd.Series([1,4,8,11,33], index=['a','b','a','b','e'])
ps3
ps3['a']

ps2['b':'d']

ps3
ps3.loc['a']
ps3.iloc[1]
ps3.iloc[1:3]

ps4 = pd.Series([33,44,77,11,99,88])
ps4

ps4>70

ps4[ps4>70]

ps4
ps4[(ps4>40) & (ps4<80)]



import pandas as pd

course = pd.Series(['BTech','MTech', 'MBA', 'BBA'])
strength = pd.Series([100,200,300,250])
fees = pd.Series([2.5, 3, 3.5,2])

course
strength
fees


pd1 =pd.DataFrame([course,strength, fees])
pd1

pd2 = pd.DataFrame({'course':course, 'strength':strength, 'fees':fees})
pd2


pd2.index

pd2.columns

pd2.dtypes
pd2

pd2.iloc[1]

pd2['course']
pd2.course


pd2

pd2.course=='MBA'

pd2.loc[pd2.course=='MBA']

pd3 = pd2.loc[pd2.strength>=200]

pd3


import pandas as pd
import numpy as np

placed = pd.Series([None,np.nan, 100, None])

placed

np.sum(placed)


course = pd.Series(['BTech','MTech','BBA','MBA'])
strength = pd.Series([100, 50, 200, 75])
fees = pd.Series([2.5, 3, 2, 4])


pd3 = pd.DataFrame({'course':course, 'strength':strength, 'fees':fees, 'placed':placed})
pd3

pd3.drop(1)

pd3.drop('course', axis=1)

pd3 = pd.DataFrame({'course':course, 'strength':strength, 'fees':fees, 'placed':placed})
pd3

pd3['course']=='MTech'
pd3[pd3['course']=='MTech']
pd3[pd3['course']=='MTech'].index
pd3.drop(pd3[pd3['course']=='MTech'].index)




pd3.strength.max()
pd3.strength.min()
pd3.strength.mean()
pd3.strength.std()
pd3.strength.sum()

pd3

np.sum(pd3.isnull())

pd3
pd3.dropna()

import pandas as pd
pd4 = pd.DataFrame([['dhiraj', 50, 'M', 10000, 2000], [None, None, None, None, None], ['kanika', 28, None, 5000, None], ['tanvi', 20, 'F', None, None], ['poonam',45,'F',None,None],['upen',None,'M',None, None]])
pd4

pd4.dropna()

pd4.dropna(axis=1)
pd4.dropna(axis='columns')
pd4.dropna(axis = 'rows')

pd4.columns

pd4
pd4[1].dropna()

pd4

pd4.dropna(axis='rows')

pd4.dropna(axis='rows', how='all' )

pd4.dropna(axis='rows', how='any' )
pd4
pd4.dropna(axis='rows', thresh=3)
pd4.dropna(axis=1, thresh=3)



pd4.fillna(0)

pd4.fillna('A')

placed= pd.Series([1,2, None, 5, None, None, 8])
placed

placed.fillna(method='ffill')
placed.fillna(method='bfill')





grades1 = {'subject1': ['A1','B1','A2','A3'],'subject2': ['A2','A1','B2','B3']   }

grades1

df1 = pd.DataFrame(grades1)
df1

grades2 = {'subject1': ['A1','B1','A2','A3'],'subject4': ['A2','A1','B2','B3']}

df2 = pd.DataFrame(grades2)
df2

df1
df2


pd.concat([df1,df2])

pd.concat([df1,df2], axis=0)

df1

pd.concat([df1,df2], axis=1)

pd.concat([df1,df2], axis=0, ignore_index=True)



import pandas as pd
#Join
rollno = pd.Series(range(1,1000001))
rollno

l1 = []
for i in range(1,1000001):
    s = 'Student ' + str(i)
    l1.append(s)
l1


["Student"+str(i) for i in range(1,11) if i%2==0]

["Student" + str(i) for i in range(1,11)]


name = pd.Series(["student" + str(i) for i in range(1,1000001)])
name



genderlist  = ['M','F']

import random

gender = random.choices(genderlist, k=1000000)
len(gender)

random.choices(population=genderlist,weights=[0.4, 0.6],k=10)


import numpy as np
#numpy.random.choice(items, trials, p=probs)
np.random.choice(a=genderlist, size=10, p=[.2,.8])




import numpy as np
marks1 = np.random.randint(40,100,size=1000000)
marks1



pd5 = pd.DataFrame({'rollno':rollno, 'name':name, 'gender':gender, 'marks1':marks1})
pd5

pd5.to_csv('synth.csv')

df = pd.read_csv('synth.csv')
df
































































