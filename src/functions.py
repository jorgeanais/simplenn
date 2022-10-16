"""
Activation funcions for neurons
"""
import numpy as np


sigm = (lambda x: 1 / (1 + np.exp(-x)), lambda x: x * (1 - x))