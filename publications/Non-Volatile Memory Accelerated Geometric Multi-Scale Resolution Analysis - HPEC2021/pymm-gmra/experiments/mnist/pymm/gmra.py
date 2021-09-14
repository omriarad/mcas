# SYSTEM IMPORTS
from tensorflow.keras.datasets import mnist
from typing import Set
from tqdm import tqdm
import argparse
import numpy as np
import os
import sys
import torch as pt
import time
import pymm


_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.abspath(os.path.join(_cd_, "..", "..", ".."))]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_
print(sys.path)


# PYTHON PROJECT IMPORTS
from mcas_gmra import CoverTree, DyadicTree
from pysrc.trees.wavelettree import WaveletTree

def main() -> None:
    init_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("covertree_path", type=str,
                        help="path to serialized json file")
    parser.add_argument("-p", "--shelf_path", type=str, default="/mnt/pmem0",
                        help="path for the pymm shelf")
    parser.add_argument("-s", "--shelf_size_mb", type=int, default=1024,
                        help="size of shelf (in mb)")
    args = parser.parse_args()

    shelf = pymm.shelf("mnist_gmra", size_mb=args.shelf_size_mb, pmem_path=args.shelf_path)

    print("loading data")
    start_time = time.time()
    (X_train, _), (X_test, _) = mnist.load_data()
    X = np.vstack([X_train, X_test])
    shelf.X = X.reshape(X.shape[0], -1).astype(np.float32)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    if not os.path.exists(args.covertree_path):
        raise ValueError("ERROR: covertree json file does not exist at [%s]"
                         % args.covertree_path)

    print("loading covertree from [%s]" % args.covertree_path)
    start_time = time.time()
    cover_tree: CoverTree = CoverTree(args.covertree_path)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    print("constructing dyadic tree")
    start_time = time.time()
    dyadic_tree = DyadicTree(cover_tree)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))

    print("constructing wavelet tree")
    start_time = time.time()
    wavelet_tree = WaveletTree(dyadic_tree,
                               shelf.X,
                               0,
                               shelf.X.shape[-1],
                               shelf=shelf)
    end_time = time.time()
    print("done. took {0:.4f} seconds".format(end_time-start_time))
    print("took script {0:.4f} seconds to run".format(end_time-init_time))


if __name__ == "__main__":
    main()

