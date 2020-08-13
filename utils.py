# Deep Learning Basics, Programming Assignment #1
# Miika Toikkanen

import numpy as np
import matplotlib.pyplot as plt

class DataGenerator():

    def __init__(self, mu, sigma):
        '''This class generates a dataset with the paramters given in mu and sigma
        Each argument holds arrays for each output class in the given order'''
        self.mu = mu
        self.sigma = sigma
        self.n_classes = len(mu)

    def sample(self, n_samples=1):
        '''Sample n_samples datapoints'''
        # Create an array of labels at random
        labels = np.random.randint(self.n_classes, size=n_samples)

        # Generate samples from the corresponding distributions
        X = []
        Y = []
        for label in labels:
            # random sample x
            x = np.random.multivariate_normal(mean=self.mu[label],cov=self.sigma[label])
            # one-hot encoded label y
            y = np.eye(1, self.n_classes, k=label)[0]
            X.append(x)
            Y.append(y)

        return np.array(X).T, np.array(Y).T


def compute_accuracy(y_hat, y):
    '''This function computes the accuracy based in the prediction y_hat and the label y'''
    # Get indices of the predictions
    pred_class = np.argmax(y, axis=0)
    real_class = np.argmax(y_hat, axis=0)
    assert len(pred_class) == len(real_class) > 0

    # Compute accuracy as percentage
    accuracy = sum(pred_class == real_class) / len(pred_class) * 100
    return accuracy
