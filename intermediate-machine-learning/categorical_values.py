import pandas as pd

d = {'col1':[1,2],'col2':[2,3],'col3':['task1','task2']}
df = pd.DataFrame(d)
print(df)

print(df.dtypes)

s = (df.dtypes == 'object')
object_cols = list(s[s].index)

#Label Encoding
from sklearn.preprocessing import LabelEncoder
label_x_train = df.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    label_x_train[col] = label_encoder.fit_transform(df[col])
print(label_x_train)

# One hot encoding
from sklearn.preprocessing import OneHotEncoder
oh_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
oh_encoded_cols = pd.DataFrame(oh_encoder.fit_transform(df[object_cols]))
oh_encoded_cols.index = df.index
removed_cat_cols = df.drop(object_cols,axis=1)
oh_x_train = pd.concat([removed_cat_cols,oh_encoded_cols],axis=1)
print(oh_x_train)