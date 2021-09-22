# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS


def make_covertree(Adj_Ms: List[np.ndarray],
                   X: np.ndarray,
                   base: float = 2) -> None:
    for pt_idx, pt in enumerate(X):
        pt_l2_dists: np.narray = ((X - pt.reshape(1,-1))**2).sum(axis=1)**(1/2)
        for scale, Adj_M in enumerate(Adj_Ms):
            Adj_M[pt_idx,:] = pt_l2_dists <= (2**scale)

