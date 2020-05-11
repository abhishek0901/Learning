import pandas as pd

#Read csv
df = pd.read_csv('read_file.csv')
print(df.col5)
# group by column5 Same as value counts
print(df.groupby('col5').col5.count())

#Min value of col3 for each name
print(df.groupby('col5').col3.min())

#group by multiple columns
print(df.groupby(['col5','col1']).col3.min())

#agg method -> lets u run multiple functions
print(df.groupby('col5').col2.agg([len,min,max,sum]))

new_df = df.groupby(['col5','col3']).col2.agg([len,min,max,sum])
print(new_df)

new_df = new_df.reset_index()
print(new_df)

#Sorting
print(df.sort_values(by='col2'))
#Sorting descending
print(df.sort_values(by = 'col2',ascending=False))
#Sorting based on index
print(df.sort_index())
#Sorting based on multiple columns
print(df.sort_values(['col5','col2']))