# SYSTEM IMPORTS
from tensorflow.keras.datasets import mnist
from typing import Set
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import os
import sys

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from gmra.trees.covertree import CoverTree, CoverTreeNode
from gmra.trees.dyadiccelltree import DyadicCellTree


def touch_dataset(root: CoverTreeNode,
                  scale: int) -> Set[int]:
    C = [root]
    idxs = {root.pt_idx}
    for s in range(scale+1, -1,-1):
        print(s, len(C))
        C_next = sum([c.get_children(s) for c in C], [])
        idxs.update([c.pt_idx for c in C_next])
        C = C_next
    return idxs

def main() -> None:
    (X_train, _), (X_test, _) = mnist.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])
    X = X.reshape(X.shape[0], -1)

    """
    max_l2_norm: float = 0
    for i in tqdm(range(X.shape[0]), desc="computing max scale"):
        l2_dist: float = np.max(((X[i] - X[i+1:,:])**2).sum(axis=1)**(1/2))
        if max_l2_norm < l2_dist:
            l2_dist = max_l2_norm
    max_scale: int = np.ceil(np.log2(max_l2_norm))
    print("max_scale: ", max_scale)
    """

    max_scale = 13

    covertree = CoverTree(max_scale=max_scale).insert(X)
    covertree.check_tree()
    print(covertree.num_pts, X.shape[0])
    print(covertree.min_scale, covertree.max_scale)

    print(len(touch_dataset(covertree.root, covertree.max_scale)))
    print(len(covertree.node_registry))

    print(len(set([n.pt_idx for n in covertree.node_registry])))
    print(all([n.parent is not None or n == covertree.root
               for n in covertree.node_registry]))

    """
    celltree = DyadicCellTree().from_covertree(covertree)
    celltree.check_tree()
    print("num cells", celltree.num_nodes)
    """


if __name__ == "__main__":
    main()

