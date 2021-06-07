# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:32:14 2021

@author: vikas
"""

#Dictionary

#Key Value Pair

stud = {'rollno':1, 'name':'vikas', 'class':'Phd'}


car = { 'brand':'Honda', 'model': 'Jazz', 'year' : 2017}
car

#add

car['color'] ='Black'
car

car['color'] ='White'
car

car['color1'] ='Black'
car
#Access

car['brand']

car['year']

car.get('year')


for i in car:
    print(i)

for i in car.values():
    print(i)
    
for i in car.keys():
    print(i)

car.keys()
car.values()


#Not Indexed
#car[0]

car.items()


for i in car.items():
    print(i)


for k,v in car.items():
    print(k, v, sep='-')

car
car.pop('model')

car
car.popitem()

car.clear()
del car



import pandas as pd
df = pd.read_csv('advertising.csv')
df.columns

l1 = list(df['TV'])

l1.sort()
l1
















