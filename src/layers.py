from typing import Callable
import numpy as np

ActivationFunction = Callable[[np.ndarray], np.ndarray]

class NeuralLayer():
    def __init__(self, n_conn: int, n_neur: int, act_f: tuple[ActivationFunction, ActivationFunction]):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur) * 2 - 1  # bias between -1 and 1
        self.W = np.random.rand(n_conn, n_neur) * 2 - 1  # weights between -1 and 1


def create_nn(topology: list[int], act_f: ActivationFunction) -> list[NeuralLayer]:
    """
    Create a neural network from a topology and an activation function
    """
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(NeuralLayer(topology[l], topology[l + 1], act_f))
    return nn