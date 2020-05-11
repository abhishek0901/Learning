import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn import tree

train_data = pd.read_csv('decision_tree_sample.csv')
print(train_data.head(3))

Y = train_data.mpg
train_data.drop(['mpg','maker'],axis=1,inplace=True)
X = train_data

model = RandomForestClassifier()
model.fit(X,Y)

print("Number of Trees Used : {}".format(model.n_estimators))

predict_train = model.predict(X)
print("Traget on train dataset : ",predict_train)

acuracy_train = accuracy_score(Y,predict_train)
print("Accuracy score on train dataset : {}".format(acuracy_train))

print("Feature Importance")
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

fn=['cylinders','displacement','horsepower','weight','acceleration','modelyear']
cn=['Bad','OK','Good']
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (5,2), dpi=200)
for index in range(0, 5):
    tree.plot_tree(model.estimators_[index],
                   feature_names = fn,
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_trees.png')