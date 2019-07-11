from sklearn import svm
import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np


# Importing the dataset
dataset = pd.read_excel('0_data.xls')
X = dataset.iloc[1:, :-1].values

# encoding categorical data
# encoder_x = LabelEncoder()
# X[:,index] = encoder_x.fit_transform(X[:,index])
# Marriage attribute converted
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# sex attribute converted
onehotencoder = OneHotEncoder(categorical_features=[4])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# education attribute converted
onehotencoder = OneHotEncoder(categorical_features=[5])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)



# Importing the dataset
dataset = pd.read_excel('data2.xls')
X1 = dataset.iloc[1:, :-1].values

# encoding categorical data
# encoder_x = LabelEncoder()
# X[:,index] = encoder_x.fit_transform(X[:,index])
# Marriage attribute converted
onehotencoder = OneHotEncoder(categorical_features=[3])
X1 = onehotencoder.fit_transform(X1).toarray()
X1 = X1[:, 1:]
# sex attribute converted
onehotencoder = OneHotEncoder(categorical_features=[4])
X1 = onehotencoder.fit_transform(X1).toarray()
X1 = X1[:, 1:]
# education attribute converted
onehotencoder = OneHotEncoder(categorical_features=[5])
X1 = onehotencoder.fit_transform(X1).toarray()
X1 = X1[:, 1:]

y = dataset.iloc[1:, -1].values


# Feature Scaling
sc = StandardScaler()
X1 = sc.fit_transform(X1)

params = {'nu': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
          'kernel': ['rbf'],
          'gamma': ['auto']}

model = svm.OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
model.fit(X)

y_pred = model.predict(X1)

y_pred = (y_pred < 0.5)
y = (y > 0.5)

cm = confusion_matrix(y, y_pred)

print cm

print("accuracy: ", (float(cm[0][0]+cm[1][1])/float(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100)

# for z in ParameterGrid(params):
#     print(z)
#     model = svm.OneClassSVM()
#     model.set_params(**z)
#     model.fit(X)
#
#     y_pred = model.predict(X1)
#
#     y_pred = (y_pred < 0.5)
#     y = (y > 0.5)
#
#     cm = confusion_matrix(y, y_pred)
#
#     print("accuracy: ", (float(cm[0][0]+cm[1][1])/float(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))*100)
