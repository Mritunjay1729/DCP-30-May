# -*- coding: utf-8 -*-
import pandas as pd
pd.set_option('display.max_columns',15)
#others - max_rows, width, precision, height, date_dayfirst, date_yearfirst

import pandas as pd
import numpy as np

#read data



df = pd.read_csv(r'D:\ML-Lab\Earlier_Batches\CBAP_13DEC\Practice_Excersices\Denco Data\20denco.csv')
#see properties of data
df.head()
df.describe()
df.columns

len(df)

df.dtypes

df['partnum']=(df['partnum']).astype('str')

df.dtypes


df1 = df.describe()

df= df1

df.shape
df.dtypes

df.head()

'''
df['region'].unique()
df['region'] = df['region'].astype('category')
df['region'].unique()
'''

df.head()


df.region.value_counts()
df.region.value_counts().plot(kind='bar')

#do summary actions
#Who are the most loyal Customers - Improve repeated sales, Target customers with low sales Volumes
#count by customers, sort, pick top 5 

df.columns

#numpy method of counts
type(df.custname.value_counts())
df.custname.value_counts().sort_values(ascending=False).head(5)


#pandas way
df.groupby('custname').size()
df.groupby('custname').size().sort_values(ascending=False)
#pickup first 5
df.groupby('custname').size().sort_values(ascending=False).head(5)
#these are most loyal customers


#extra- ??
'''
df.groupby(['custname'])['margin'].nlargest(3)


df.sort_values(['revenue'], ascending= True).groupby( 'region' ).mean()
'''

#Which customers contribute the most to their revenue - How do I 
#retain these customers & target incentives
#find total revenue of each customer, sort in descending.

df.groupby('custname').aggregate({'revenue':np.sum})
df.groupby('custname').aggregate({'revenue':np.sum}).sort_values(by='revenue', ascending=False)
df.groupby('custname').aggregate({'revenue':np.sum}).sort_values(by='revenue', ascending=False).head(5)

#these are top 5 customers who gave revnue to Denco

#extras ???

'''
df.groupby('custname').aggregate({'revenue':[np.sum, 'size'] }).sort_values(by='revenue', ascending=False) #error

df[['revenue','custname']].groupby('custname').agg['size'] 
.sort_values(by='revenue', ascending=False) #error
'''




#What part numbers bring in to significant portion of revenue - 
#Maximise revenue from high value parts

#grouby partnum, find value of revenue, sell these more

df[['partnum','revenue']].sort_values(by='revenue', ascending=False)
df[['partnum','revenue']].sort_values(by='revenue', ascending=False).head(5)



#these are top parts which give max revenue
#if total sales has to be considered
df[['partnum','revenue']].groupby('partnum').sum()
df[['partnum','revenue']].groupby('partnum').sum().sort_values(by='revenue', ascending=False).head(5)
#this total revenue value giving items

gdf = df[['partnum','revenue']].groupby(['partnum'], as_index=False).agg([sum, np.mean])

gdf.to_csv("gdf.csv", index=False)

gdf = pd.read_csv("gdf.csv")
gdf

gdf=gdf[2:]
gdf = gdf.rename(columns = {'Unnamed: 0':'Partnum', 'revenue': 'revenue_Sum', 'revenue.1': 'revenue_mean'}).reindex()
gdf


gdf.sort_values(by='revenue_mean', ascending=False).head(5)
df.dtypes



#What parts have the highest profit margin - What parts are driving profits & 
#what parts need to build further
#check for margin value, their individual margin and total sales margin like revenue

df[['partnum','margin']].sort_values(by='margin', ascending=False)
df[['partnum','margin']].sort_values(by='margin', ascending=False).head(5)
df[['partnum','margin']].sort_values(by='margin', ascending=False).tail(5)

#these are top parts which give max margins
#if total sales has to be considered


pd.set_option('display.max_columns',1000)


df[['partnum','margin']].groupby('partnum')['margin'].sum()
df1= df[['partnum','margin']].groupby('partnum').sum().sort_values(by='margin', ascending=False).head(5)

df1

df1.to_csv('aa.csv')



#this total revenue value giving items


#Most sold items

df.groupby('partnum').size().sort_values(ascending=False).head(5)


#which regions gave max revenue
df2 = df[['revenue', 'region']].groupby( 'region').sum().sort_values(by='revenue', ascending=False).head(5)
df2



#East gave max revenue
df[['revenue', 'region']].groupby( 'region').sum().sort_values(by='revenue', ascending=False).plot(kind='barh')

