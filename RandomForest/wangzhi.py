import numpy as np
import pandas as pd
import scipy
import sklearn.ensemble as ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import sklearn.tree as tree
import sklearn.metrics as metrics

df = pd.read_csv("train_10000.csv")
df = df.fillna(df.mean())
df.head()
print(df.head())
y = df["label"]
x = df.iloc[:, 1:-1]

test = pd.read_csv("validate_1000.csv")
test = test.fillna(test.mean())

y1 = test["label"]
x1 = test.iloc[:, 1:-1]

# x = df.drop("label", axis=1)


seed = 5
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)
param_grid = {"criterion": ["entropy", "gini"],
              "max_depth": [14],
              "n_estimators": [30, 40],
              "max_features": [0.25, 0.3, 0.4],
              "min_samples_split": [4, 6, 8]}
rfc = ensemble.RandomForestClassifier()
rfc_cv = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=4)
# rfc_cv.fit(x_train, y_train)
# predict_test = rfc_cv.predict(x_test)
# print("随机森林")
# print(metrics.classification_report(predict_test, y_test))
# rfc_cv.best_params_
# print(rfc_cv.best_params_)
rfc2 = ensemble.RandomForestClassifier(criterion="entropy", max_depth=14, min_samples_split=4, n_estimators=40)
rfc2.fit(x_train, y_train)
predict_test = rfc2.predict(x1)
print(predict_test)
print("随机森林")
print(metrics.classification_report(predict_test, y1))