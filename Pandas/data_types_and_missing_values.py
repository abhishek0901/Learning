import pandas as pd

data = pd.read_csv('read_file.csv')
print('Print Data type of a particular column 2 : {}'.format(data.col2.dtype))
print('The data type of all columns \n',data.dtypes)
print('Converting col2 datatype tp float64')
data.col2 = data.col2.astype('float64')
print(data)
print('Print DataType of Index : {}'.format(data.index.dtype))
print(data[pd.isnull(data.col1)])
print("Fill Na value with some number")
print(data.col1.fillna('23'))
print("Replace Name3 with Name7 in col5")
print(data.col5.replace('name3','name7'))