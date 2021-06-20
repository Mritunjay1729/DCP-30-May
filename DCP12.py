# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:46:56 2021

@author: vikas
"""

import pandas as pd
import numpy as np

rollno = pd.Series(range(1,1001))

rollno

name = pd.Series(["student" + str(i) for i in range(1,1001)])
name

genderlist  = ['M','F']

import random
#gender = random.choices(genderlist, k=1000)
gender= np.random.choice(a=genderlist, size=1000,replace=True, p=[.6,.4])
gender


marks1 = np.random.randint(40,100,size=1000)

marks2 = np.random.randint(40,100,size=1000)

fees = np.random.randint(50000,100000,size=1000)

fees.mean()

course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000, p=[0.4, 0.5,0.09,0.01])

course

city = np.random.choice(a=['Delhi', 'Gurugram','Noida','Faridabad'], size=1000, replace=True, p=[.4,.2,.2,.2])


course = np.random.choice(a=['BBA','MBA','BTECH', 'MTech'], size=1000, p=[0.4, 0.5,0.09,0.01])
pd8 = pd.DataFrame({'rollno':rollno, 'name':name, 'course':course, 'gender':gender, 'marks1':marks1,'marks2':marks2, 'fees':fees,'city':city})
pd8

pd8.describe()

pd8.count()

pd8.columns
pd8['gender'].value_counts()  #if col has spaces

pd8['course'].value_counts()  #if col has spaces


pd8.groupby('course').size()

pd8.groupby('course').count()



categ = ['course', 'gender','city']

pd8.head(2)

pd9 = pd8[categ]

pd9

pd8.groupby(['gender','city','course']).size()
pd8.columns
pd8.groupby(['city','course'])['fees'].sum()

pd8.columns
#pd8.groupby('marks1').aggregate(min, max)
pd8.columns
pd81 = pd8.groupby('course').agg({"marks1": ["min","mean", "max", "size", "std"],
                           "marks2": ["min","mean", "max", "size", "std"]})

pd81.to_csv('gp1.csv')

pd91 = pd8.groupby('course', as_index=True).agg({"marks1": ["min","mean", "max", "size", "std"],
                           "marks2": ["min","mean", "max", "size", "std"]})
pd91.to_csv('gp2.csv')

pd91.plot(kind = 'barh')


pd8.to_excel("data.xlsx",sheet_name='pd8', index=False)
pd91.to_excel("data.xlsx",sheet_name='pd91')


with pd.ExcelWriter('data.xlsx') as writer:
    pd8.to_excel(writer, sheet_name='first', index=False)
    pd91.to_excel(writer, sheet_name='second')
    
    
 




