from builtins import zip
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Perceptron(object):
    """
    Perceptron classifier

    Parameters
    -----------
    learning_rate: float
        Learning rate (between 0.0 and 1.0)
    epochs: int
        Passes (epochs) over the training set.

    Attributes
    ----------
    w_dot : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassification in every epochs
    """
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        """
        Fit method for training data
        :param X: {array-like}, shape = [n_samples, n_features]
                Training vectors, where 'n_samples' is the number of samples,
                'n_features' is the number of features
        :param y: {array-like}, shape = [n_samples]
                Target values
        :return: self : object
        """
        n_features = X.shape[1]
        self.w_dot = np.zeros(shape=(1+n_features))
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for x_i, y_i in zip(X, y):
                y_i_hat = self.predict(x_i)
                # the learning rule implemented: w <- w + α(y — f(x))x
                update = self.learning_rate * (y_i - y_i_hat)
                # update bias weight
                self.w_dot[0] += update
                # update features weights
                self.w_dot[1:] = self.w_dot[1:] + update * x_i
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate the net input
        :param X: sample data
        :return: the dot product of features and weights
        """
        return np.dot(X, self.w_dot[1:]) + self.w_dot[0]

    def predict(self, X):
        """
        Return class label after unit step.
        :param X: Sample data
        :return: Classified label, with activate function is sign function
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


means = [[2, 2], [4, 4]]
covariance = [[1, 0], [0, 1]]
N = 500
class_num = 2

X_0 = np.random.multivariate_normal(means[0], covariance, N)
X_1 = np.random.multivariate_normal(means[1], covariance, N)

X = np.concatenate((X_0, X_1), axis=0)
y = np.array([[-1]*N, [1]*N]).reshape(class_num*N,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

perceptron_obj = Perceptron(epochs=100)
perceptron_obj.train(X_train, y_train)
# prediction or labeling
y_hat = perceptron_obj.predict(X_test)
# Calculate the accuracy of the classification
print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_hat)))
# Visualize X_test on 2D graph, colored by y_test
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_hat)
plt.show()
