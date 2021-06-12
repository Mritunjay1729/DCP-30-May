# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:51:42 2021

@author: vikas
"""

# Tuple


#Hetro...

t1 = (1,3,"vikas", True, 4.5)
t1

type(t1)


e = 1,3,4

type(e)



#indexing
t1[0]

#Unmutable
t1[0]=120

t1.remove()
#Error


'''
if (condition)
{
 

}


if (condition):
    Statement1
    Statement2
'''

a = 10
b =20


if (a == b):
    print("a is equal to b")
    print("a is equal to b")
    print("a is equal to b")


print()
print("a is equal to b")
    
    
a <= b
a >= b



a != b

a=30
b=20

if (a<b):
    print(str(a) + " is less than " + str(b))
else:
    print(str(a) + " is greater than " + str(b))
    
    

marks = int(input("Enter Marks -> "))

if (marks>80):
    print("A")
elif (marks<=80 and marks>70):
    print("B")
elif (marks>60 and marks<=70):
    print("C")
else:
    print("Fail")




#Looping 



l1 = []

l1.append(int(input('Enter value-> ')))
l1.append(int(input('Enter value-> ')))
l1.append(int(input('Enter value-> ')))
l1.append(int(input('Enter value-> ')))
l1.append(int(input('Enter value-> ')))
l1.append(int(input('Enter value-> ')))
l1.append(int(input('Enter value-> ')))
l1.append(int(input('Enter value-> ')))


l1 = []

cnt=0

while (cnt<10):
    print(cnt)
    cnt = cnt + 1




r1 = range(1,11)

for i in r1:
    print(i)


for i in (r1):
    print(i**2)

'''
2 * 1 = 2
2 * 2 = 4
2 * 10 = 20
'''

t = 2
for i in range(1,11):
    print('{0} * {1} = {2}'.format(t,i,t*i))
    

t = 3
for i in range(1,11):
    print('{0} * {1} = {2}'.format(t,i,t*i))
    
t = 4
for i in range(1,11):
    print('{0} * {1} = {2}'.format(t,i,t*i))
    

#Nested Loops


for t in range(1,11):
    for i in range(1,11):
        print('{0} * {1} = {2}'.format(t,i,t*i))
 

t = 4
for i in range(1,11, 2):
    i=i+1
    print('{0} * {1} = {2}'.format(t,i,t*i))
 

l1 = [3, 5, 1, 8, 6]

for i in l1:
    print(i)


teamA = ['India', 'Australia','Pakistan', 'England']

for i in teamA:
    print(i)

'India' in teamA


for i in teamA:
    if (i == 'India'):
        print(True)



for i in teamA:
    if i == 'Pakistan' :
        print(i , "Inner")
        break
    print(i, "Outer")
    
    
for i in teamA:
    if i == 'Australia_A' :
        print(i , "Inner")
        continue
    print(i, "Outer")
   

for i in teamA:
    if i == 'Australia_A' :
        print(i , "Inner")
        pass
    print(i, "Outer")
    
    
    

l1 = [3,6,1,2,9,8,7]

for i in l1:
    print(i)
    if (i==6):
        print("yes")
        #break

teamA

for i in teamA:
    if i == 'Pakistan' :
        print(i , "Inner")
        break
    print(i, "Outer")
    
    
for i in teamA:
    if i == 'Pakistan' :
        print(i , "Inner")
        continue
    print(i, "Outer")
   


l1 = [3,6,1,2,9,8,7]

for i in l1:
    print(i)
    if (i==6):
        print("yes")
        #break


a=10
b=20


print(a+b)
print(a-b)
print(a*b)
print(a/b)


print(a+b)
print(a-b)
print(a*b)
print(a/b)


print(a+b)
print(a-b)
print(a*b)
print(a/b)


print(a+b)
print(a-b)
print(a*b)
print(a/b)


def oper():
    print(a+b)
    print(a-b)
    print(a*b)
    print(a/b)


oper()
oper()




def oper1(a,b):
    print(a+b)
    print(a-b)
    print(a*b)
    print(a/b)


x=10
y=20
oper1(x,y)

x=30
y=40
oper1(x,y)
oper1(30,70)


def welcome(fname, lname):
    print("hello "+fname + lname)


welcome("ABC", 'CSE')


welcome("CDE")

l1=[]

def data(name, age=18, email='none'):
    l1.append((name,age,email))
    
data('a', 17, 'a@gmail.com')

l1

data('b', 21, 'b@gmail.com')

l1

data('c', 22)

l1



data('d')

l1



name = input("Enter name")
age= int(input('Enter age'))
email = input("Enter Email")
data(name, age, email)


l1


data('cc', )

def oper2(a,b):
    add = a+b
    sub = a-b
    mul = a*b
    div = a/b
    print(add, sub, mul, div)
    return (add, sub, mul, div)


a,b,c,d = oper2(30,40)

print(a,b,c,d)



















































































































































































