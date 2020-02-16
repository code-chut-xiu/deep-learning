import numpy as np


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Initialize the weight vector and bias scalar with zeroes
def initialize_weight_bias_with_zeroes(dimension):
    return np.zeros((dimension, 1)), 0


