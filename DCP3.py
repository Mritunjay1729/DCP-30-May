# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 18:40:26 2021

@author: vikas
"""

x=3

print(x)

print(type(x))

print (type(3.7))

x =x +1
print(x)

print(x+1)

print(x)

print(x*2)

print(x-2)

print (x/3)
print (x%3)

x=11

x/3

x%3

x =int(input("Enter Number "))

if (x%2 == 0):
    print("Even")
else:
    print("Odd")

x = x + 1
x

x += 1
x

x *= 2
x

x **= 2
x


y =x**2
y

x=5

print(x, x+1, x-2, x*2, x/3)


#Boolean

t = True
f = False

t
f

type(t)


marks = int(input("Enter Marks  "))
course = int (input("Enter 0 for not done /  Enter 1 for done"))

if (marks >= 75 and course==1):
    print("Allowed")
else:
    print("Not allowed")

AND
A B Y
0 0 0
0 1 0
1 0 0
1 1 1



marks = int(input("Enter Marks  "))
course = int (input("Enter 0 for not done /  Enter 1 for done"))

if (marks >= 75 or course==1):
    print("Allowed")
else:
    print("Not allowed")
    
    
OR
A B Y
0 0 0
0 1 1
1 0 1
1 1 1


NOT
A Y
0 1
1 0

not (marks >= 75)
not(False)


t = True
f = False
t1 =True

print(t and f)
print ( t and t1)

print (t or f)

print (f and f)


#String

fname = "Vikas"
type(fname)

lname ='Khullar'

name = fname + lname 

name = fname + " " + lname
print(name)

len(name)

print ("don't")

h = 'hello'
w = 'world'

print(len(h))

print(h.capitalize())

type(h)
h.capitalize()
h.upper()
h.lower()

h.center(15)
h.rjust(15)
h.ljust(15)

s = "Welcome to Java"

s.replace('Java', 'Python')
s.replace('a', 'e')


s = input('Enter Yr Name  ')

len(s)
s

s.strip()





















































































