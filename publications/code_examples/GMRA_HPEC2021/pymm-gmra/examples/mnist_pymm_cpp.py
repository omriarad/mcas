# SYSTEM IMPORTS
from tensorflow.keras.datasets import mnist
from typing import Set
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import numpy as np
import os
import pymm
import sys
import torch as pt

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from gmra_trees import CoverTree, DyadicTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_examples", type=int, default=10000,
                        help="number of examples from mnist to process")
    parser.add_argument("-p", "--shelf_path", type=str, default="/mnt/pmem0",
                        help="path for the pymm shelf")
    parser.add_argument("-s", "--shelf_size_mb", type=int, default=1024,
                        help="size of shelf (in mb)")
    args = parser.parse_args()

    shelf = pymm.shelf("mnist_gmra", size_mb=args.shelf_size_mb, pmem_path=args.shelf_path)

    (X_train, _), (X_test, _) = mnist.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])
    shelf.X = pt.from_numpy(X.reshape(X.shape[0], -1).astype(np.float32))

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

    cover_tree = CoverTree(max_scale=max_scale)
    print(cover_tree.num_nodes, shelf.X.shape[0])
    print(cover_tree.min_scale, cover_tree.max_scale)

    for pt_idx in tqdm(list(range(shelf.X.shape[0]))[:args.num_examples],
                       desc="building covertree"):
        cover_tree.insert_pt(pt_idx, shelf.X)
        # print()

    print(cover_tree.num_nodes, shelf.X.shape[0])
    print(cover_tree.min_scale, cover_tree.max_scale)

    root_idxs = cover_tree.root.get_subtree_idxs(cover_tree.max_scale)
    print("number of pts accessible from root: ", root_idxs.size(),
          "expected number of pts accessible (num pts processed): ", args.num_examples)

    dyadic_tree = DyadicTree(cover_tree)
    print(dyadic_tree.num_nodes, dyadic_tree.root.idxs.size())
    print(dyadic_tree.validate())

    print(dyadic_tree.num_levels)

    for level in range(dyadic_tree.num_levels):
        print(dyadic_tree.get_idxs_at_level(level))


if __name__ == "__main__":
    main()

