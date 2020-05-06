import pandas as pd
from numpy import NaN
from sklearn.impute import SimpleImputer
x_train = [[0.1,0.2,0.3,NaN,0.5],[0.1,0.2,0.3,0.1,0.5],[0.1,0.2,0.3,0.2,0.5]]
my_imputer = SimpleImputer()
x_train_imputed = pd.DataFrame(my_imputer.fit_transform(x_train))
print(x_train_imputed)