import pandas as pd

#Read csv
df = pd.read_csv('read_file.csv')
#Describe summary of given column
print(df.col2.describe())

#Describe summary of string data
print(df.col5.describe())

#Returning just mean of particluar column
print(df.col2.mean())

#Returning list of unique values in a list
print(df.col5.unique())

#returning list of frequency of all unique values
print(df.col5.value_counts())

#Making mean 0 with map()
r_m = df.col2.mean()
print(df.col2.map(lambda p:p - r_m))

#Same operation with apply method
def r_p(row):
    if row.col5 == 'name4':
        row.col3 = 10
    return row

print(df.apply(r_p,axis='columns'))

#which row has maximum value of col2
print(df.col1.idxmax())