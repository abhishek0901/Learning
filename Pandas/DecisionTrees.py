from sklearn import tree
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

data = pd.read_csv('decision_tree_sample.csv')
feature_cols = ['cylinders','displacement','horsepower','weight','acceleration','modelyear']
Y = data.mpg.to_numpy()
data.drop(['mpg','maker'],axis=1,inplace=True)
X = data.to_numpy()
#create model
model = tree.DecisionTreeClassifier(criterion='gini')
#Train model
model.fit(X,Y)

plt.figure(figsize=(30,10))
a = plot_tree(model,
              feature_names=feature_cols,
              class_names=['Bad','OK','Good'],
              filled=True,
              rounded=True,
              fontsize=5)

b=1