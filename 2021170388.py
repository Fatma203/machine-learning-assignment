import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


#read data
data = pd.read_csv('assignment2dataset.csv')
print(data.head())


#encoding for Extracurricular Activities column
le = preprocessing.LabelEncoder()
data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

print(data.head())


print(data.corr())

#Features
x=data.iloc[:,0:2]
#dependent variable
y=data['Performance Index']


#scaling data using standardization
y= (y-y.mean())/y.std()

x= (x-x.mean())/x.std()


#transform data
x_transformed = pd.DataFrame({'1': [1] * 10000})


def transform(deg):
    for i in range(1, deg + 1):
        x_transformed['a^' + str(i)] = x['Hours Studied'] ** i
        x_transformed['b^' + str(i)] = x['Previous Scores'] ** i

    for j in range(2, deg):
        x_transformed['a^' + str(j) + 'b'] = x['Hours Studied'] ** j * x['Previous Scores']
        x_transformed['ab^' + str(j)] = x['Hours Studied'] * x['Previous Scores'] ** j

    for z in range(1, deg):
        if (z + z > deg):
            break
        else:
            x_transformed['a^' + str(z) + 'b^' + str(z)] = x['Hours Studied'] ** z * x['Previous Scores'] ** z

    return x_transformed



degree=2
x_transform=transform(degree)

x_train, x_test, y_train, y_test = train_test_split(x_transform, y, test_size = 0.20,shuffle=True,random_state=10)


#apply linear regression model

poly_model = LinearRegression()
poly_model.fit(x_train, y_train)

y_train_pred = poly_model.predict(x_train)
y_test_pred = poly_model.predict(x_test)

# Step 4: Calculate the Mean Squared Error for both training and testing sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("Training Mean Squared Error ", mse_train)
print("Testing Mean Squared Error", mse_test)











