import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

from src.layers import create_nn
from src.functions import sigm
from src.process import train
from src.cost import l2_cost


# Create a dataset
n = 500  # number of samples
p = 2  # number of features

X, y = make_circles(n_samples=n, factor=0.5, noise=.05)
y = y[:, np.newaxis]

# Create a neural layer
topology = [p, 4, 8, 1]
neuronal_net = create_nn(topology, sigm)


# Train the neural network
loss = []

for i in range(10000):
    pY = train(neuronal_net, X, y, l2_cost, lr=0.02)

    if i % 100 == 0:
        loss.append(l2_cost[0](pY, y))
    
        res = 50
        _x0 = np.linspace(-1.5, 1.5, res)
        _x1 = np.linspace(-1.5, 1.5, res)
        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neuronal_net, np.array([[x0, x1]]), y, l2_cost, train=False)[0][0]

        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")

        plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c="skyblue")
        plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c="salmon")
        
        plt.savefig(f"plots/plot_{i}.png")
        plt.close()
        plt.clf()

        plt.plot(range(len(loss)), loss)
        plt.savefig(f"plots/loss_{i}.png")
        plt.close()
        plt.clf()