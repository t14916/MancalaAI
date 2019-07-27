import numpy as np
import scipy.special as sp


class NeuralNetwork:

    """
    NOTE: This class, and this project as a whole draws heavily from the book Make Your Own Neural Network by Tariq
    Rashid, but implements a slightly more complicated neural network which supports more than three layers.

    NOTE: This version of Neural Network is tuned to train for Mancala.
    """
    def __init__(self, layers,  learning_rate=0.5):
        """"
        initializer for neuralNetwork class, requires learning rate and a variable number of layers, which
        which are given by list layers for which the length of the list represents the # of layers
        and each list represents the layer.
        """

        self.learning_rate = learning_rate
        # layers is a list of layers, with the first being the input layer and the final being the output layer
        # Each element in layers refers to the number of weights in said layer.
        # NOTE: cannot be a np array (Those do not support jagged arrays natively
        self.layers = layers

        # Enumerate(**) creates an enumerated list of (index, val) tuples up to and including the second to
        # last value of the list (last value of layers doesn't matter because it is the output, no weight matrix)
        # Whole list is a list of tuples of (layer_length, nextLayer_length)
        # to help create weight matrices easily
        layer_next = [(layer, self.layers[index + 1]) for index, layer in enumerate(self.layers[:len(self.layers) - 1])]

        # link weight matrix, w_i_j for weight from node i to node j in the next layer
        # row : element of the next layer (i)
        # column : element of the current layer (j)
        # list of np matrices which contain weights, as described above
        # weights are determined randomly along a normal distribution around 0 w/ std dev
        # 1/sqrt(len_column)
        self.weights = [np.random.normal(0.0, pow(l[0], -0.5), (l[1], l[0])) for l in layer_next]

        # sets the activation function
        # primarily for testing purposes(we can change this later)
        self.activation_function = lambda x: sp.expit(x)

    def query(self, input_list):
        """
        Runs through the whole matrix and given an input, returns an output given the current weight matrix.
        Throws an illegal exception if the input is of the incorrect size.
        """

        # Checks input size
        if len(input_list) != self.layers[0]:
            raise Exception('Input has incorrect size. The size of input was {}, but should be {}.'
                            .format(len(input_list), self.layers[0]))

        # converts input to np array
        prev_layer = np.array(input_list, ndmin=2).T

        # First list is the input
        layers = [input_list]
        for weight_layer in self.weights:

            # print(prev_layer)

            # multiplies previous layer with the weights to get the next layer
            next_layer = np.dot(weight_layer, prev_layer)

            # applies activation function on the next layer
            next_layer = self.activation_function(next_layer)

            # appends the next layer
            layers.append(next_layer)
            # sets previous layer as next layer
            prev_layer = next_layer

        # Last list in layers should be the output.
        return layers[-1]

    def train(self, input_list, target_list):
        """
        Runs through the whole matrix and given an input, and then compares output to target. Then compares output
        to target_list and backpropogates the error. Throws an illegal exception if the input or target is of the
        incorrect size.
        """

        # Checks input size
        if len(input_list) != self.layers[0]:
            raise Exception('Input has incorrect size. The size of input was {}, but should be {}.'
                            .format(len(input_list), self.layers[0]))

        # Checks target size
        if len(target_list) != self.layers[-1]:
            raise Exception('Input has incorrect size. The size of input was {}, but should be {}.'
                            .format(len(target_list), self.layers[-1]))

        # Converts target to np array
        targets = np.array(target_list, ndmin=2).T
        # Converts input to np array
        prev_layer = np.array(input_list, ndmin=2).T

        # converts target to target
        # First list is the input
        layers = [np.array([input_list]).T]
        for weight_layer in self.weights:

            # multiplies previous layer with the weights to get the next layer
            next_layer = np.dot(weight_layer, prev_layer)

            # applies activation function on the next layer
            next_layer = self.activation_function(next_layer)

            # appends the next layer
            layers.append(next_layer)
            # sets previous layer as next layer
            prev_layer = next_layer

        # Last list in layers should be the output.
        output = layers[-1]

        # Finds the Output error
        output_error = targets - output

        weights_len = len(self.weights)
        # loop to create a list of numpy arrays of errors in reverse order (i.e. output error is first error)
        error_array = []
        for i in range(0, weights_len):
            if i == 0:
                # first error is output error
                error_array += [output_error]
            else:
                # print(error_array)
                error_array += [np.dot(self.weights[weights_len - i].T, error_array[i - 1])]
        #print(error_array)
        # Backpropogation loop

        # b = np.dot((error_array[0] * layers[2] * (1.0 - layers[2])), np.transpose(layers[1]))
        # self.weights[1] += self.learning_rate * b
        # b = np.dot((error_array[1] * layers[1] * (1.0 - layers[1])), np.transpose(layers[0]))
        # self.weights[0] += self.learning_rate * b

        for i in range(0, weights_len):
            # Applies backpropogation function to weights matrix
            # print(len(error_array[-i - 1]))
            # print(len(layers[-i - 1]))
            a = error_array[i] * layers[-i - 1] * (1.0 - layers[-i - 1])
            prop_dot_product = np.dot(a, np.transpose(layers[weights_len - i - 1]))
            self.weights[-i - 1] += self.learning_rate * prop_dot_product
