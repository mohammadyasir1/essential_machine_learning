import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./content/placement.csv')
# print(df.head())
# plt.scatter(df['cgpa'], df['package'])
# plt.xlabel('CGPA')
# plt.ylabel('Package(LPA)')
# plt.show()

X = df.iloc[:, 0:1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# creating object of LinearRegression
lr = LinearRegression() 

# to train the data
lr.fit(X_train, y_train)

print(lr.predict(X_test.iloc[0].values.reshape(1,1)))
# print(X_test)
# print()
# print(y_test)

print(df.head())
plt.scatter(df['cgpa'], df['package'])
plt.plot(X_train, lr.predict(X_train), color='r')
plt.xlabel('CGPA')
plt.ylabel('Package(LPA)')
plt.show()

m = lr.coef_
b = lr.intercept_
print(m)
print(b)

# y = mx+b
# print(m*7.82+b)