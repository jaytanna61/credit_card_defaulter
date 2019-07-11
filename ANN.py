# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_excel('Data2.xls')
X = dataset.iloc[1:, :-1].values
y = dataset.iloc[1:, -1].values

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train=y_train.astype('int')

# Building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# Init ANN
classifier = Sequential()

# Add i/p and 1st hidden layer
classifier.add(Dense(output_dim = 28 , init = 'uniform', activation = 'relu', input_dim= 28))
classifier.add(Dropout(p=0.2))

# Adding Hidden Layers
classifier.add(Dense(output_dim = 28 , init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim = 28 , init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.2))

# Add o/p layer
classifier.add(Dense(output_dim = 1 , init = 'uniform', activation = 'sigmoid'))

# compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = classifier.fit(X_train, y_train, batch_size= 10, nb_epoch=500)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_test1 = (y_test > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1, y_pred)

print cm


# Plots model
from keras.utils import plot_model
plot_model(classifier, to_file='model.png')

for layer in classifier.layers:
    weights = layer.get_weights()
    print weights
