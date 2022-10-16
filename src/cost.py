import numpy as np

l2_cost = (
    lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
    lambda Yp, Yr: (Yp - Yr)
    )