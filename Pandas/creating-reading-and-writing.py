#This file talks about basics in Pandas
import pandas as pd
#Q: How to create DataFrame

'''
Simplest way to create DataFrame is to use dictionary
'''

dc = {'col1':[1,2,3],'col2':[2,3,4],'col3':['val1','val2','val3']}

df = pd.DataFrame(dc)

print(df)

'''
If we want to have indexes
'''

df = pd.DataFrame(dc,index = ['row1','row2','row3'])

print(df)

'''
reading from csv file
'''

df = pd.read_csv('read_file.csv')

print(df)

'''
make col3 as index
'''

df = pd.read_csv('read_file.csv',index_col='col3')
print(df)

'''
Dealing with series in pandas
'''

df = pd.Series([1,2,3],index=['t1','t2','t3'],name='Sample')
print(df)

'''
Saving dataframe to csv
'''

df.to_csv('output.csv')

'''
reading it back
'''
df = pd.read_csv('output.csv',index_col=0)
print(df)