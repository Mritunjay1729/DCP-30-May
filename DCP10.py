# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 18:52:06 2021

@author: vikas
"""
import numpy as np

xn1 = np.random.randint(10, 30, size=10)
xn1


np.mean(xn1)

np.std(xn1)

xn1 = np.array([55.444, 3.2222, 6.57666])
xn1

xn1.round(2)




np.floor([1.2, 1.6])

np.ceil([1.2, 1.6])

np.trunc([1.2, 1.6])

np.round([1.2, 1.6])

np.trunc([-1.2, -1.6])

np.floor([-1.2, -1.6])

np.round([-1.23434, 1.654455],2)

np.round([1.234,2.368],2)


np.full((3,4), 2)

np.full((3,4), np.pi)


#concatenate arrays
x4=np.array([1,2,3,4,5,6,7])
x4.size
#3 x 4
x5 = np.zeros(5)
x5
x4

x4b=np.concatenate([x4,x5])
x4b

'''
x4c=np.concatenate([x4,np.zeros(3 * 4 - x4.size)])
x4c
'''


x=np.arange(6).reshape(2,3)
x
y=np.arange(10,16).reshape(2,3)
y

np.concatenate([x,y], axis=0)

np.concatenate([x,y], axis=1)



x=np.arange(10,20)
x

np.split(x,5)

x=np.arange(10,19)

y=x.reshape([3,3])
y


#upper and lower
upper, lower = np.vsplit(y,[2])
upper  #first 2 rows
lower # last row

y

left, right = np.hsplit(y,[3])
left
right

x=np.random.randint(10,100, size=(3,6))
x 


x.min()

x.min(axis=1) #min in each row

x.max(axis=0) #min in each col

x
np.median(x)  #median values in full dataset
np.max(x)  #max

'''
what is the difference between x.min() and np.min(x) ?
'''

type(x)

x.min()
type(x)
np.min(x)



x=np.arange(0, 10000000)

len(x)

ser =9999999

for i in x:
    if(i==ser):
        print("searched")
        break
    print("searching")


ser in x


x=np.array([30,49,50,60, 49])

np.equal(x, 49) #all values equal to 48


np.sum(np.equal(x,49))




z = np.random.randint(0, 100, size=10)
z

sum(np.equal(z,43))

np.greater(z, 40) #values greater than 40


np.sum(np.greater(z,40))  #how many values > 40


sum(np.greater(x,40))
np.less(z, 50)  #values < 50
sum(np.less(z, 50))

np.greater_equal(x, 40)  #values >= 40



x=np.random.randint(10, size=(3,4))
x



np.all(x >= 0)


np.any(x > 10)



x=np.random.randint(10, 100, size=(3,4))
x

sum(sum(np.greater(x,50)))

np.any(x>99)

x
np.sum(x > 90)
np.sum(x > 90, axis=1)

np.sum(x > 90, axis=0)

















