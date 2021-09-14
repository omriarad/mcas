# SYSTEM IMPORTS
from tensorflow.keras.datasets import mnist
from typing import Set
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_examples", type=int, default=10000,
                        help="number of examples from mnist to process")
    parser.add_argument("-s", "--max_scale", type=int, default=-1,
                        help="max scale, if -1 determine automatically")
    args = parser.parse_args()

    (X_train, _), (X_test, _) = mnist.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])
    X = X.reshape(X.shape[0], -1).astype(np.float32)
    X = X[:min(args.num_examples, X.shape[0])]
    X_pt = pt.from_numpy(X)
    print(X.shape)

    if args.max_scale <= 0:
        max_l2_norm: float = 0
        for i in tqdm(range(X.shape[0]-1), desc="computing max scale"):
            l2_dist: float = np.max(((X[i] - X[i+1:,:])**2).sum(axis=1)**(1/2))
            if max_l2_norm < l2_dist:
                max_l2_norm = l2_dist

        print(max_l2_norm)
        args.max_scale = int(np.ceil(np.log2(max_l2_norm)))
        print("max_scale: ", args.max_scale)

    cover_tree = CoverTree(max_scale=args.max_scale)

    for pt_idx in tqdm(list(range(X_pt.shape[0]))[:args.num_examples],
                       desc="building covertree"):
        cover_tree.insert_pt(pt_idx, X_pt)
        # print()

    print(cover_tree.num_nodes, X.shape[0])
    print(cover_tree.min_scale, cover_tree.max_scale)

    print(cover_tree.validate(X_pt))
    print(cover_tree.parent_vector())

    cd = os.path.abspath(os.path.dirname(__file__))
    save_path = os.path.join(cd, "tree.json")
    cover_tree.save(save_path)

    new_tree: CoverTree = CoverTree(save_path)
    print(new_tree.num_nodes, X.shape[0])
    print(new_tree.min_scale, new_tree.max_scale)

    print(new_tree.validate(X_pt))
    print(new_tree.parent_vector())

    print(new_tree == cover_tree)


if __name__ == "__main__":
    main()

