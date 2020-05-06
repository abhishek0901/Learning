#Indexing
import pandas as pd
df = pd.read_csv('read_file.csv')
print(df.col1)
print("OR")
print(df['col1'])
#Accessing specific value
print(df['col1'][1])
#Selecting Particular row from data frame
print(df.iloc[6])
#getting specific rows all 1 column
print(df.iloc[[2,4],1])
#what if in case of last 3 values
print(df.iloc[-3:])
#Name based value access
print(df.loc[:,['col1']])
#Get col3 of 3rd row
print(df.loc[3,'col3'])
#Biggest use case of .loc is when data is indexed
df = pd.read_csv('read_file.csv',index_col='col4')
print(df)

#Setting another column as index
df = df.set_index('col5')
print(df)

#print all rows where col1 = 2
print(df.loc[df.col1 == 2])

#multiple conditions
print(df.loc[(df.col1 == 2) & (df.col3 > 3)])

#Similarly Or condition
print(df.loc[(df.col1 == 2) | (df.col1 == 1)])

#Build features
print(df.loc[df.col2.isin([1,3])])
print(df.loc[df.col3.isnull()])
print(df.loc[df.col3.notnull()])
