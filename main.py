# Deep Learning Basics, Programming Assignment #1
# Miika Toikkanen

from utils import DataGenerator, compute_accuracy
import numpy as np
import math
import matplotlib.pyplot as plt

class Layer():
    def __init__(self):
        '''Base class for the network layers.
        Each layer holds its own gradient, input, output and name in memory for later use.
        Methods are implemented for the layers that need them.'''
        self.layer_grad = 0
        self.layer_in = 0
        self.layer_out = 0
        self.layer_name = ''

    def forward(self, x):
        pass
    
    def backward(self, x):
        pass

    def update_parameters(self, lr):
        pass

    def __str__(self):
        return self.layer_name

class Input(Layer):
    def __init__(self, n_in):
        '''Input layer. Values and gradients simply flow through.'''
        super().__init__()
        self.n_in = n_in
        self.layer_name = 'Input layer (%d)' % n_in

    def forward(self, x):
        self.layer_out = x
        return self.layer_out
    
    def backward(self, y):
        self.layer_grad = y
        return self.layer_grad

class ReLU(Layer):
    def __init__(self):
        '''Rectified Linear Unit activation function'''
        super().__init__()
        # Datatype needs to be forced as float, to avoid integer outputs
        # The max-function needs to be vectorized to apply on the whole array.
        self.vect_max = np.vectorize(max, otypes=[np.float64])
        self.layer_name = 'ReLU'

    def forward(self, x):
        '''Apply vecotrized version of max(0,x).'''
        self.layer_in = x     
        self.layer_out = self.vect_max(0, x)
        return self.layer_out
    
    def backward(self, y):
        '''Gradient at the layer input side is dZ = dA * g'(Z), 
        where for ReLU g'() = g().
        y contains dA and self.layer_out contains the output from forward pass.'''
        self.layer_grad = np.multiply(y, self.layer_out)
        return self.layer_grad

class FC(Layer):
    def __init__(self, n_in, n_out, mu=0, sigma=0.001):
        '''A fully connected layer'''
        super().__init__()
        self.W = []
        self.dW = []
        self.b = 0
        self.db = 0
        self.n_in = n_in
        self.n_out = n_out
        self.initialize_weights(mu, sigma)
        self.layer_name = 'Fully connected layer (%d, %d)' % (n_in, n_out)

    def initialize_weights(self, mu, sigma):
        '''Initialize weights randomly with gaussian distribution. bias is set to 0.'''
        self.W = np.random.normal(loc=mu, scale=math.sqrt(sigma), size=(self.n_out, self.n_in))
        self.b = 0

    def forward(self, x):
        '''Multiply the input with weights.
        row of ones is augmented to the input, 
        while the broadcasted bias column is augmented to weights.'''
        self.layer_in = x
        # Augment a row of ones to the input
        ones = np.ones((1, x.shape[1]))
        full = np.full((self.W.shape[0], 1), self.b)
        x_aug = np.concatenate((x, ones), axis=0)
        W_aug = np.concatenate((self.W, full), axis=1)
        # Dot product with bias term included
        self.layer_out = np.matmul(W_aug, x_aug)
        return self.layer_out
    
    def backward(self, dZ):
        '''The gradient on the input side is W * dZ, 
        Gradient for W is W = 1/m * dZ * A_previous, 
        Gradient for b is db = 1/m * sum(dZ).
        '''
        assert dZ.shape[1] != 0
        A_previous = self.layer_in
        self.dW = np.matmul(dZ, A_previous.T)
        self.layer_grad = 1/dZ.shape[1] * np.matmul(self.W.T, dZ)
        self.db = 1/dZ.shape[1] * np.sum(dZ, axis=1, keepdims=True)
        return self.layer_grad

    def update_parameters(self, lr):
        '''Apply the previously computed gradients to weights'''
        self.W -= lr * self.dW
        self.b -= lr * self.db

class MSELoss(Layer):
    def __init__(self):
        '''Mean Squared Error loss.'''
        super().__init__()
        self.layer_name = 'MSE loss'

    def forward(self, y_hat, y):
        '''Compute the MSE error and gradient from prediction y_hat and label y'''
        self.layer_in = y_hat
        squared = (y_hat - y)**2
        summed = np.mean(squared, axis=0)
        N = y_hat.shape[0]
        self.layer_out = 1/N * summed
        self.layer_grad = y_hat - y
        return self.layer_out
        
    def backward(self, x):
        '''Equal to pred - label, Computed in forward_pass method.'''
        return self.layer_grad

class MLP():
    def __init__(self, mu=0, sigma=0.001):
        '''A Multi-Layer Perceptron model.
        
        Layer configuration is defined here.'''
        # Define the layers
        self.layers = []
        self.layers.append(Input(2))
        self.layers.append(FC(2, 10, mu=mu, sigma=sigma))
        self.layers.append(ReLU())
        self.layers.append(FC(10, 10, mu=mu, sigma=sigma))
        self.layers.append(ReLU())
        self.layers.append(FC(10, 2, mu=mu, sigma=sigma))
        self.loss = MSELoss()
        
        # Print to terminal
        print('Model architechture:')
        for idx, layer in enumerate(self.layers + [self.loss]):
            print('Layer %d: %s' % (idx, layer))
            
    def forward_propagate(self, x):
        '''Propagate input through all layers.
        
        Returns the final value as a prediction.'''
        for layer in self.layers:
            x = layer.forward(x)
        y_hat = x
        return y_hat

    def backward_propagate(self, y_pred, y_label):
        '''Compute loss and propagate back through all layers.'''
        loss = self.loss.forward(y_pred, y_label)
        cost = loss.mean()
        grad = self.loss.backward(loss)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return cost

    def optimize(self, lr):
        '''Update all layers that have parameters.'''
        for layer in self.layers:
            layer.update_parameters(lr)

if __name__ == "__main__":
    
    # fix seed
    np.random.seed(111)

    # Parameters
    # Data
    train_size = 100
    test_size = 100
    mu0 = [1, 1]
    mu1 = [-1, -1]
    sigma0 = [[1, 0], [0 ,1]]
    sigma1 = [[1, 0], [0 ,1]]
    # Weights
    mu = 0
    sigma = 0.001
    # Training
    lr = 0.1
    epochs = 1000
    batch_size = 25

    # Generate data
    G = DataGenerator([mu0, mu1], [sigma0, sigma1])
    train_x, train_y = G.sample(train_size)
    test_x, test_y = G.sample(test_size)

    # Define network
    net = MLP(mu=mu, sigma=sigma)

    # Compute initial accuracy
    test_y_hat = net.forward_propagate(test_x)
    initial_acc = compute_accuracy(test_y_hat, test_y)
    print('accuracy before training: %.02f %%' % initial_acc)

    # Train model
    metrics = []
    n_batches = train_x.shape[1]//batch_size
    print('number of batches: ', n_batches)
    for epoch in range(epochs):
        # For plotting
        # Compute metrics
        train_y_hat = net.forward_propagate(train_x)
        train_loss = net.backward_propagate(train_y_hat, train_y)
        test_y_hat = net.forward_propagate(test_x)
        test_loss = net.backward_propagate(test_y_hat, test_y)
        train_accuracy = compute_accuracy(train_y_hat, train_y)
        test_accuracy = compute_accuracy(test_y_hat, test_y)
        #print([train_loss, test_loss, train_accuracy, test_accuracy])
        metrics.append([train_loss, test_loss, train_accuracy, test_accuracy])

        # Split data into batches
        batches_x = np.array_split(train_x, n_batches, axis=1)
        batches_y = np.array_split(train_y, n_batches, axis=1)
        # Optimize on each batch
        for batch_x, batch_y in zip(batches_x, batches_y):    
            batch_y_hat = net.forward_propagate(batch_x)
            net.backward_propagate(batch_y_hat, batch_y)
            net.optimize(lr)

    # Compute final accuracy
    test_y_hat = net.forward_propagate(test_x)
    final_acc = compute_accuracy(test_y_hat, test_y)
    print('accuracy after %d epochs: %.02f %%' % (epochs, final_acc))

    # Plot metrics
    metrics = np.array(metrics)
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_ylabel('loss')
    ax[0].plot(metrics[:, 0])
    ax[0].plot(metrics[:, 1])
    ax[0].legend(['train','test'])
    #ax2 = ax1.twinx()
    ax[1].set_ylabel('acc [%]')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylim(0, 100)
    ax[1].plot(metrics[:, 2])
    ax[1].plot(metrics[:, 3])
    plt.show()
