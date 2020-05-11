import pandas as pd
df = pd.read_csv('read_file.csv')
print(df)

#If we want to change the name of col5 to name_columns
print(df.rename(columns={'col5':'name_columns'}))

#If we want to rename index:0 and index:1 as start_1 and start_2
print(df.rename(index={0:'start_1',1:'start_2'}))

#If we want to update row axis name to samples and column axis to features
print(df.rename_axis('samples',axis = 'rows').rename_axis('features',axis ='columns'))