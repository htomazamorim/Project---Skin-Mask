import numpy as np

class NeuralNetwork(object):
    def __init__(self, vecLayers, learningRate=0.5):
        """
            Definition of neural network.
            :param vecLayers: vector that contains how many neurons there is in each layer
            :param learningRate: learningRate factor for gradient descent method
        """

        # Setting learning rate
        self.lr = learningRate

        # W - weigt matrix, b - bias matrix
        self.W = []
        self.b = []

        # Initialisation of Weight matrix
        for i in range(1, len(vecLayers)):
            self.W.append( np.random.uniform(-1/np.sqrt(vecLayers[i-1]), 1/np.sqrt(vecLayers[i-1]), (vecLayers[i], vecLayers[i-1])) )

        # Initialisation of Bias matrix
        for i in range(1, len(vecLayers)):
            self.b.append( np.zeros((vecLayers[i], 1)) )

    def changeLearnRate(self, lr):
        # Method for setting another learning rate
        self.learningRate = lr

    def sigmoid(self, x):
        # Learning function set as a Sigmoid
        return 1/(1+np.exp(-x))

    def sigmoid_dev(self, x):
        # Sigmoid derivative
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def forwardProp(self, input):
        """
            This function is responsible for the forward propagation,
            calculating the values of each layer

            :param input: input of neural network
            :return: last neural network layer vector
        """

        # Z - layers vector withour sigmoid 'squishification'
        self.Z = []
        # a -layers after sigmoid application
        self.a = []

        # Calculating first layer
        # Z0 = W0 * INPUT + b0
        self.Z.append( np.dot(self.W[0], input) + self.b[0] )
        self.a.append( self.sigmoid (self.Z[0]) )

        # Calculating layers
        # Z(i) = W(i) * a(i-1) + b(i)

        for i in range(1, len(self.W)):
            self.Z.append( np.dot(self.W[i], self.a[i-1]) + self.b[i] )
            self.a.append( self.sigmoid(self.Z[i]) )

        return self.a[ len(self.a)-1 ]

    def calculateFinalLayer(self, aL, y, input):
        self.delta_L = (aL - y)*self.sigmoid_dev(self.Z[ len(self.Z)-1 ])
        self.gradW = []
        self.gradB = []

        # Case where the neural network does not have hidden layers
        # then the gradient is calculated using the input itself
        if(len(self.W) == 1):
            self.gradW.append(np.dot(self.delta_L, input.reshape(1,-1)))
        else:
            # Gradient calculated using previous layer
            self.gradW.append(np.dot(self.delta_L, (self.a[len(self.a) - 2]).reshape(1, -1) ))
        self.gradB.append( self.delta_L )

    def calculateHiddenLayers(self, W_after, delta_after, a_k_1, L):
        # Calculating hidden layers
        self.delta_bef = np.dot(W_after.T, delta_after)*self.sigmoid_dev(self.Z[L])
        self.gradB.append ( self.delta_bef )
        self.gradW.append ( np.dot(self.delta_bef, a_k_1.reshape(1, -1)) )

    def backwardProp(self, aL, y, input, batchSize):
        """
            # Calculating final layer in order to start backpropagation

            :param aL: last neural layer
            :param y: waited output data
            :param input: neural network input
            :param batchSize: sample batch's size to compute the gradient
            :return: no return
        """

        self.calculateFinalLayer(aL, y, input)

        ## REVER ESSA PORRA
        if(len(self.W) >= 2):
            # Regressive loop counting
            # obs: start in w(len(w)-1) because last layer is already calculated above
            for i in range(len(self.W)-2, 0, -1):
                self.calculateHiddenLayers(self.W[i+1], self.gradB[(len(self.W)-1) - i - 1], self.a[i-1], i)

            self.calculateHiddenLayers(self.W[1], self.gradB[(len(self.W)-1)-1], input, 0)

        # Descendent gradient method
        for i in range(len(self.W)):
            self.W[i] -= (self.lr)*( (1/batchSize) * self.gradW[(len(self.W) - 1) - i] )
            self.b[i] -= (self.lr)*( (1/batchSize) * self.gradB[(len(self.b) - 1) - i] )

    def train(self, input, y, overBatch, batchSize, k):
        if(overBatch):
            # Training for a batch of samples

            s=0

            for i in range(batchSize):
                # Putting y and x into column form
                self.yT = y[i].reshape(-1, 1)
                self.xT = input[i].reshape(-1, 1)

                aL = self.forwardProp(self.xT)
                self.backwardProp(aL, self.yT, self.xT, batchSize)

                if(k%10 == 0):
                    # Cost function calculated only for the points when we wat to plot
                    # In this case, it at each 10 interactions
                    s += np.square(aL - self.yT)

            if (k % 10 == 0):
                # Return the current value of the cost function
                return np.mean(s)/batchSize
            else:
                # Only returns an actual value if necessary
                return 0

        else:
            # Training for just one sample
            a = self.forwardProp(input.reshape(-1,1))

            self.backwardProp(a,y.reshape(-1,1),input.reshape(-1,1), batchSize)

            return np.mean(np.square(a - y))


    def saveNN(self):
        # Saving W and b in matrices
        np.save('./W', self.W, allow_pickle=True)
        np.save('./b', self.b, allow_pickle=True)

    def loadNN(self):
        # Loading W and b from directory
        self.W = np.load('./W.npy', allow_pickle=True)
        self.b = np.load('./b.npy', allow_pickle=True)

"""
Train the neural network over the data X given
and the known answers y.
"""
def trainNNoverData(NN, X, y, lr, bs, maxIter):
    # Setting learning rate for Neural Network
    NN.changeLearnRate(lr)

    # output data size
    samples = len(y)

    # vector for cost function values
    cost = []
    # x_axis for graph to be plotted later
    x_axis = []

    #
    c2 = 1;

    p = 0

    # Iteration counter
    j = 0

    while (c2 > 10 ** -3 and j < maxIter):
        # Condition for not exiting before time
        c2 = 1

        # Puts X and y in random orders
        random_idxs = np.random.choice(len(y), len(y), replace=False)
        X_shuffled = X[random_idxs, :]
        y_shuffled = y[random_idxs]

        ## Check shuffle
        i = 0

        while ((bs * i + bs) != (samples - samples % bs)):
            # Increasing cost function
            p = NN.train(X_shuffled[bs * i:bs * i + bs], y_shuffled[bs * i:bs * i + bs], True, bs, j)

            # Calculating cost function over all dataset
            c2 += p

            # Training for all dataset
            i += 1

        # At each 10 interactions we save the cost function value
        if (j % 10 == 0):
            x_axis.append(j)
            cost.append(c2 - 1)

        j += 1  # Increasing iteration number

        # Saving W and b matrices for each 100 interactions
        if (j % 10 == 0):
            NN.saveNN()
            print(j)

    # Saving neural network at the end
    NN.saveNN()

    return x_axis, cost

"""
This function compares the answer given by the model
with the actual known answer to evaluate training.
"""
def validationData(NN, X, y):
    c=0
    for i in range( len(X) ):
        prediction = NN.forwardProp(X[i].reshape(-1,1))

        a = prediction
        b = y[i]

        if( b == 1 and a > 0.9):
            c+=1
        elif( b == 0 and a < 0.1):
            c+=1

    print(100*c/len(X))

"""
Initializing Neural Network

Setting learning rate
lr = 0.5

Setting neural network
NN = NeuralNetwork([3, 5, 15, 1], lr)

Loading input data
X = np.load('inputData.npy')
y = np.load('outputData.npy')

Shuffling data
random_idxs = np.random.choice(len(y), len(y), replace=False)
X_shuffled = X[random_idxs, :]
y_shuffled = y[random_idxs]
"""

"""
Code below used to separate data into training and testing

# qtdSamples = 101500

X2 = X_shuffled[0:qtdSamples]
y2 = y_shuffled[0:qtdSamples]

np.save('./X_training', X2, allow_pickle=True)
np.save('./y_training', y2, allow_pickle=True)
np.save('./X_testing', X_shuffled[qtdSamples:145000], allow_pickle=True)
np.save('./y_testing', y_shuffled[qtdSamples:145000], allow_pickle=True)

X2 = np.load('./X_training.npy', allow_pickle=True)
y2 = np.load('./y_training.npy', allow_pickle=True)
Xb = np.load('./X_testing.npy', allow_pickle=True)
yb = np.load('./y_testing.npy', allow_pickle=True)
"""

"""
Code used for training and plotting error

c = 0

Pre-evaluation of mean error (Cost function)
for i in range(len(X2)):
    a = NN.forwardProp(X2[i].reshape(-1, 1))
    c += np.mean(np.square(a - y2[i]))

print('Pre-evaluation  of error: ' + str(c/len(X2)))

Getting loss over training
x_axis, cost = trainNNoverData(NN, X2, y2, lr, 300, 5000)

In case I want to load previous Network
NN.loadNN()

Validation pre-trained data
validationData(NN, X2, y2)

Plots cost function over time to see convergence
plt.plot(x_axis, cost, color='g')
plt.show()
"""



