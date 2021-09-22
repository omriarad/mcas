# SYSTEM IMPORTS
from tensorflow.keras.datasets import mnist
import pymm
import numpy as np
import os
import sys

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from gmra.trees.covertree import CoverTree


def main() -> None:
    shelf = pymm.shelf("mnist_gmra", size_mb=1024, pmem_path="/mnt/pmem0")
    (X_train, _), (X_test, _) = mnist.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])

    shelf.tree = CoverTree()
    shelf.tree.insert(X)
    print(shelf.tree.num_pts, X.shape[0])
    print(shelf.tree.min_scale, shelf.tree.max_scale)

    


if __name__ == "__main__":
    main()

