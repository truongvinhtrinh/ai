# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
# pip install --upgrade tensorflow

# Installing Keras
# pip install --upgrade keras
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

# init ANN
classifier = Sequential()
# First hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", input_dim=11, units=6))
# Second hidden layer
classifier.add(Dense(kernel_initializer="uniform", activation="relu", units=6))
# Output layer
classifier.add(Dense(kernel_initializer="uniform", activation="sigmoid", units=1))

# Compile the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN
classifier.fit(x=X_train, y=y_train, batch_size=10, epochs=100)

# Prediction
y_hat = classifier.predict(x=X_test)
y_hat = (y_hat > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_hat)


