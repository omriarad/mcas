# SYSTEM IMPORTS
from tensorflow.keras.datasets import cifar10
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
    parser.add_argument("data_dir", type=str,
                        help="directory to where covertree will serialize itself to")
    parser.add_argument("--validate", action="store_true",
                        help="if enabled, perform an expensive tree validate operation")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    print("loading data")
    # This part would be replaced with loading a custom dataset.
    # NOTE: the entire dataset does *not* have to fit into DRAM.
    #       we can use a memory-mapped version of the dataset
    #       (or some custom data loader) which will feed
    #       one example at a time to the covertree.
    # P.S.: when data is fed into the covertree it MUST be a torch tensor!!!
    (X_train, _), (X_test, _) = cifar10.load_data()
    X: np.ndarray = np.vstack([X_train, X_test])
    X = X.reshape(X.shape[0], -1).astype(np.float32)

    # this is a throwaway tensor...just used to feed into the covertree
    X_pt = pt.from_numpy(X)
    print("done")

    # NOTE: build covertree w/ max scale = ceil(log_2(max(||x_i, x_j||_2^2)))
    #       this is to ensure that every point in the dataset can be added to the tree
    #       and is reachable from the root. For Cifar10, I know beforehand that
    #       this value is 14.
    # If you DON'T know this value for your dataset, you can compute it but be WARNED
    # it will take either LOTS of ram OR LOTS of time.
    cover_tree = CoverTree(max_scale=14)

    # make a nice progress bar. the CoverTree type will accept
    # a bunch of examples at once, however it doesn't have a progress bar.
    # I know its a little slower to use a pure python loop but I'm willing
    # to make the tradeoff so I can see the ETA.
    # NOTE: you can replace this entire process with the following line:
    #   cover_tree.insert(X)
    for pt_idx in tqdm(list(range(X_pt.shape[0])),
                       desc="building covertree"):
        cover_tree.insert_pt(pt_idx, X_pt)

    if(args.validate):
        print("validating covertree...this may take a while")
        assert(cover_tree.validate(X_pt))

    filename = "cifar10_covertree.json"
    filepath = os.path.join(args.data_dir, filename)

    print("serializing covertree to [%s]" % filepath)
    cover_tree.save(filepath)


if __name__ == "__main__":
    main()

