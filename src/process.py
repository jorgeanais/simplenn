import enum
import numpy as np
from typing import Callable

from .layers import NeuralLayer

CostFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]

def train(
    neural_net: list[NeuralLayer],
    X: np.ndarray,
    y: np.ndarray,
    cost_fn: tuple[CostFunction, CostFunction],
    lr: float = 0.05,
    train: bool = True
) -> np.ndarray:
    """Train the NN"""

    out = [(None, X)]  # output of each layer, [(z,a), ...] > list[tuple[float, float]]
    
    # Forward pass
    for l, _ in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    if train:

        # Backward pass
        deltas = []  # list of deltas for each layer
        for l in range(len(neural_net) - 1, 0, -1):
            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(neural_net) - 1:
                # Compute delta for the last layer
                deltas.insert(0, cost_fn[1](a, y) * neural_net[l].act_f[1](a))
            else:
                # Compute delta for the hidden layers
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W

            # Gradient descent
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr
        
    return out[-1][1]


