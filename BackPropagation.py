from random import seed
from random import randrange
from random import random
from datetime import datetime
from csv import reader
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

    def accumulate_inputs(self):
        return self.output

    def update_neuron_weights(self, inputs, l_rate):
        for j in range(len(inputs)):
            self.weights[j] += l_rate * self.delta * inputs[j]
        self.weights[-1] += l_rate * self.delta


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

    def backward_propagate_error(self, expected_outputs):
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
                    errors.append(neuron.calculate_output_error(j, expected_outputs))
            # For each neuron, calculate the
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.calculate_delta_error(errors[j])

    def update_network_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = []
                for neuron in self.network[i-1]:
                    inputs.append(neuron.accumulate_inputs())
            for neuron in self.network[i]:
                neuron.update_neuron_weights(inputs, l_rate)

    def train(self, data, learn_rate, n_epochs, n_outputs):
        # train a certain number of times (epochs)
        for epoch in range(n_epochs):
            sum_error = 0
            # For each set of data
            for row in data:
                # Do the first run through, get the first output
                outputs = self.forward_propagate(row)
                # Initializes the outputs to all equal 0
                expected_outputs = [0 for i in range(n_outputs)]
                # Sets each of these outputs to its actual value
                expected_outputs[row[-1]] = 1
                sum_error += sum([(expected_outputs[i] - outputs[i]) ** 2 for i in range(len(expected_outputs))])
                self.backward_propagate_error(expected_outputs)
                self.update_network_weights(row, learn_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch + 1, learn_rate, sum_error))

    def predict(self, row):
        outputs = self.forward_propagate(row)
        prediction = outputs.index(max(outputs))
        # print('Expected=%d, Got=%d' % (row[-1], prediction))
        return prediction


def load_csv(my_file):
    data = list()
    with open(my_file, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data.append(row)
    return data


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def parse_csv(filename):
    dataset = load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0]) - 1)
    # normalize input variables
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    return dataset


def backprop_algorithm(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    my_network = NeuralNetwork(n_inputs, n_hidden, n_outputs)
    my_network.create_network()

    my_network.train(train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = my_network.predict(row)
        predictions.append(prediction)
    return predictions


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def run_algorithm(fold, train_set, test_set, learning_rate,
                  number_of_epochs, number_of_hidden_nodes):
    predicted = backprop_algorithm(train_set, test_set, learning_rate,
                                   number_of_epochs, number_of_hidden_nodes)
    actual = [row[-1] for row in fold]
    accuracy = accuracy_metric(actual, predicted)
    score = (accuracy)
    return score


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def initialize_algorithm(data, number_of_folds, learning_rate, number_of_epochs, number_of_hidden_nodes):
    folds = cross_validation_split(data, number_of_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted_result = run_algorithm(fold, train_set, test_set, learning_rate,
                                         number_of_epochs, number_of_hidden_nodes)
        scores.append(predicted_result)
    return scores


def main():
    seed(1)  # Makes sure the random numbers always start the same
    # seed(datetime)  # Makes sure the random numbers always start differently.
    my_dataset = parse_csv('iris.csv')
    number_of_folds = 5
    learning_rate = 0.3
    number_of_epochs = 500
    number_of_hidden_nodes = 5

    my_scores = initialize_algorithm(my_dataset, number_of_folds, learning_rate, number_of_epochs,
                                     number_of_hidden_nodes)
    print('Scores: %s' % my_scores)
    print('Mean Accuracy: %.3f%%' % (sum(my_scores) / float(len(my_scores))))


main()
