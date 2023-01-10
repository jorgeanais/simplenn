"""
Activation funcions for neurons.
Each activation function consist in a tuple of two functions:
the first one defines the activation function, meanwhile the second
one defines the derivative of the activation function.
"""
import numpy as np


sigm = (lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x))