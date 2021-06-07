# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 20:02:30 2021

@author: vikas
"""

#List, Set, Tupple, Dictionary

#Hetrogeneous or Homogeneous Data Type

'''
array 
a->int
a = [1,2,3,4]
'''

#Hetrogeneous

L1 = []
L1 = [1,2,3]

L2 = [5, True, "Vikas", 6.4]


'''
Name Rno Result
A    1   True
BB   2   False

'''

#Indexed

L2[0]
L2[1]
L2[2]
L2[3]
#L2[4]

# Mutable pr changable

L2[0] = 8.6
L2

type(L2[0])

#UnOrdered

L3 =[4,3,7,6]
L3

#Access

L3[0]
L3[1]
L3[2]
L3[3]

L3



for n in L3:
    print(n**2)
    

L4 =[1,4,9]

i = 9

for n in L4:
    if(n==i):
        print(True)
    else:
        print(False)
        
L2


for n in L2:
    print(n , type(n))


L2

type(L2[2])
L2[2].upper()

#L2[0].upper()


rng = range(10)
rng

l = list(rng)
l

rng = range(10, 100)
rng

l = list(rng)
l

rng = range(10, 100, 5)
l = list(rng)
l

rng = range(1,101)
l=list(rng)


for n in l:
    print(n)
    
l  

l[1]
l[4]

len(l)
l

l[0:10]

l[10:20]

l[:20]

l[90:]

l

L2

len(L2)

L2.append('Khullar')
L2

L2.append(True)
L2

len(L2)

print(L2.append(True))

L2

#L2 = L2.append(True)
#Error 

L2

print(L2)


L2 = [5, True, "Vikas", 6.4]


L2.append("Khullar")
L2.append("Khullar")
L2.append("Khullar")
L2.append(True)
L2
L2.append('Vikas')

#ADD Duplicate elements

L2.remove(True)
L2
L2

L2.remove("Khullar")
L2

L2.remove("Vikas")
L2

L2.remove(6.4)


L2
L2.pop()

L2
print(L2.remove('Vikas'))
L2

print(L2.pop())
L2

L2.pop(1)
L2

L2.pop(1)
L2



L2 = [5, True, "Vikas", 6.4]

del L2[1]

L2.clear()
L2

del L2

print()



rng1 = list(range(1000000000))

rng2 = list(range(1000000000))

rng1 = list(range(1000000000))


# Copy List

a=10
b=a
b

a=20
a
b


L2 = [5, True, "Vikas", 6.4]

L1 = L2

L2
L1

L2.append("111")

L2
L1

L3 = L2.copy()

L2.append("222")

L2
L3

L1 = list(range(1,11))
L1
L1.reverse()
L1


L1= [3,2,6,7,1,8]
L1.sort()
L1



fruits = ['apple', 'cherry', 'banana']
fruits
fruits.sort()
fruits

#put mango in 2nd position
fruits.insert(1, 'mango')
fruits
fruits.sort(reverse=True)
fruits














































