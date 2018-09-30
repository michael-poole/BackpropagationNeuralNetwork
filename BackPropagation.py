from random import seed
from random import random
from datetime import datetime
from math import exp


class Neuron:
    def __init__(self, inputs):
        self.inputs = inputs
        self.output = None
        self.delta = None
        self.weights = []

    def calculate_input_weights(self):
        i = 0
        while i < self.inputs + 1:
            self.weights.append(random())
            i += 1

    def print_out_neuron(self):
        weight_string = "Weights: "
        for weight in self.weights:
            weight_string = weight_string + str(weight) + ", "
        weight_string = weight_string.rstrip(", ")
        neuron_string = weight_string
        if self.output is not None:
            neuron_string += "\n\t\t\t " + "Output: " + str(self.output)
        if self.delta is not None:
            neuron_string += "\n\t\t\t " + "Delta Error: " + str(self.delta)
        return neuron_string

    def activate(self, n_weights, inputs):
        activation = n_weights[-1]  # Does the bias first
        for i in range(len(n_weights) - 1):
            activation += n_weights[i] * inputs[i]  # multiplies the initial weights by the inputs.
        return activation

    def sigmoid(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def calculate_output(self, my_inputs):
        activation = self.activate(self.weights, my_inputs)
        self.output = self.sigmoid(activation)
        return self.output

    def sigmoid_derivative(self, my_output):
        return my_output * (1.0 - my_output)

    def calculate_delta_error(self, my_error):
        self.delta = my_error * self.sigmoid_derivative(self.output)
        return self.delta

    def calculate_hidden_error(self, prev_neuron_index):
        return self.weights[prev_neuron_index] * self.delta

    def calculate_output_error(self, neuron_index, expected_outputs):
        return expected_outputs[neuron_index] - self.output


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.inputNodes = input_nodes
        self.hiddenNodes = hidden_nodes
        self.outputNodes = output_nodes
        self.network = []

    def create_network(self):
        i = 0
        hidden_layer = []
        while i < self.hiddenNodes:
            my_neuron = Neuron(self.inputNodes)
            my_neuron.calculate_input_weights()
            hidden_layer.append(my_neuron)
            i += 1
        self.network.append(hidden_layer)

        j = 0
        output_layer = []
        while j < self.outputNodes:
            my_neuron = Neuron(self.hiddenNodes)
            my_neuron.calculate_input_weights()
            output_layer.append(my_neuron)
            j += 1
        self.network.append(output_layer)

    def print_network(self):
        for layer in self.network:
            if layer == self.network[len(self.network) - 1]:
                print("Output Layer:")
            else:
                print("Hidden Layer:")
            k = 0
            for neuron in layer:
                k += 1
                weight_string = "\tNeuron " + str(k) + " "
                weight_string += neuron.print_out_neuron()
                print(weight_string)

    def forward_propagate(self, input_layer):
        inputs = input_layer
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                new_inputs.append(neuron.calculate_output(inputs))
            inputs = new_inputs
        return inputs

    def backward_propagate_error(self, expected):
        # Start with the output and work backwards (Must be reversed)
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            # If it is not the output layer, must accumulate error from following layer
            if i != len(self.network) - 1:
                # For each neuron
                for j in range(len(layer)):
                    error = 0.0
                    # Accumulate the error from all of its output neurons
                    for neuron in self.network[i + 1]:
                        error += neuron.calculate_hidden_error(j)
                    errors.append(error)
            # If it is the output layer
            else:
                # For each neuron, the error is the expected output minus the actual output
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron.calculate_output_error(j, expected))
            # For each neuron, calculate the
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.calculate_delta_error(errors[j])


seed(1)  # Makes sure the random numbers always start the same.
# seed(datetime.now())  # Makes sure the random numbers are different each time we start
myNetwork = NeuralNetwork(2, 3, 2)
myNetwork.create_network()
output = myNetwork.forward_propagate([1, 0])
expected = [0, 1]
myNetwork.backward_propagate_error(expected)
myNetwork.print_network()
