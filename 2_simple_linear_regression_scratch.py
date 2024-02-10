class LR:
    # linear regression from scratch
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = 0
        den = 0

        for i in range(X_train.shape[0]):
            num = num + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + (X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean())

        self.m = num/den
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self, X_test):
        return self.m * X_test + self.b

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./content/placement.csv')
# print(df.head())
X = df.iloc[:, 0].values
y = df.iloc[:, 1].values
print(X)
print()
print(y)

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
print(X_train.shape)

lr = LR()
print(lr.fit(X_train, y_train))

print(X_test[0])

print(lr.predict(X_test[0]))
