# SYSTEM IMPORTS
from typing import List, Tuple, Union
from scipy.linalg import qr
import numpy as np


# PYTHON PROJECT IMPORTS
# from gmra_trees import DyadicTree


WaveletNodeType = "WaveLetNode"


def mindim(sigmas: np.ndarray,
           errortype: str,
           err: float) -> int:
    s2: float = np.sum(sigmas**2)

    tol: float = None
    if errortype.lower() == "absolute":
        tol = err**2
    else:
        tol = err*s2

    dim: int = 0
    while dim < sigmas.shape[0]-1 and s2>tol:
        dim += 1
        s2 = s2 - sigmas[dim]**2
    return dim


def rand_pca(A: np.ndarray,
             k: int,
             its: int = 2,
             l: int = None,
             shelf=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U: np.ndarray = None
    s: np.ndarray = None
    V: np.ndarray = None

    if l is None:
        l = k + 2

    n, m = A.shape
    if (its*l >= m/1.25) or (its*l >= n/1.25):
        U, s, V = np.linalg.svd(A, full_matrices=False)

        U = U[:, :k]
        s = s[:k]
        V = V[:, 1:k]

    else:
        H: np.ndarray = None
        if n >= m:
            if shelf is None:
                H = A.dot(2*np.random.randn(m, l) - np.ones(m, l))
            else:
                shelf.rand = np.random.randn(m, l)
                H = A.dot(2*shelf.rand - (shelf.ndarray((m,l), dtype=float)*0+1))

            F: np.ndarray = None
            if shelf is None:
                F = np.zeros(n, its*l)
            else:
                F = shelf.nparray((n, its*l), dtype=float)*0
            F[:n, :l] = H

            for it in range(its):
                H = H.T.dot(A).T
                H = A.dot(H)
                F[:n, (it+1)*l:(it+2)*l] = H

            Q,_,_ = qr(F, mode="enconomic")
            U2, s, V = np.linalg.svd(Q.T.dot(A))
            U = Q.dot(U2)

            U = U[:, :k]
            s = s[:k]
            V = V[:,:k]

        else:
            if shelf is None:
                H = (2*np.random.randn(n, l) - np.ones(n, l)).dot(A).T
            else:
                shelf.rand = np.random.randn(n,l)
                H = (2*shelf.rand - (shelf.ndarray((n,l), dtype=float)*0+1)).dot(A).T

            F: np.ndarray = None
            if shelf is None:
                F = np.zeros(m, its*l)
            else:
                F = shelf.nparray((m, its*l), dtype=float)*0
            F[:n, :l] = H
            F[:m, :l] = H

            for it in range(its):
                H = A.dot(H)
                H = H.T.dot(A).T
                F[:m, (it+1)*l:(it+2)*l] = H

            Q,_,_ = qr(F, mode="enconomic")
            U, s, V2 = np.linalg.svd(A.dot(Q))
            V = Q.dot(V2)

            U = U[:, :k]
            s = s[:k]
            V = V[:,:k]

    return U, s, V


def node_function(X: np.ndarray,
                  manifold_dim: int,
                  max_dim: int,
                  is_leaf: bool,
                  errortype: str = "relative",
                  shelf=None,
                  threshold: float = 0.5,
                  precision: float = 1e-2) -> Tuple[np.ndarray, int, float,
                                                 np.ndarray, np.ndarray]:
    mu: np.ndarray = np.mean(X, axis=0, keepdims=True)

    X_mean_centered: np.ndarray = X - mu
    radius: float = np.sqrt(np.max((X_mean_centered**2).sum(axis=-1)))

    size: int = max(1, X.shape[0])

    sigmas: np.ndarray = None
    basis: np.ndarray = None
    if is_leaf or manifold_dim == 0:
        V, s, _ = rand_pca(X_mean_centered, min(min(X_mean_centered.shape), max_dim))
        rem_energy: float = max(np.sum(np.sum(X_mean_centered**2) - np.sum(s**2)), 0)
        sigmas = np.hstack([s, [np.sqrt(rem_energy)]]) / np.sqrt(size)

        dim: int = None
        if not is_leaf:
            dim = min(s.shape[0], mindim(sigmas, errortype, threshold))
        else:
            dim = min(s.shape[0], mindim(sigmas, errortype, precision))
        basis = V[:, :dim]

    else:
        V, s, _ = rand_pca(X_mean_centered, min(min(X_mean_centered_shape), manifold_dim))
        sigmas = s / np.sqrt(size)
        if V.shape[-1] < manifold_dim:
            V = np.hstack([V, np.zeros((V.shape[0], manifold_dim - size))])
        basis = V[:, :min(manifold_dim, int(np.sum(sigmas > 0)))]

    return mu, X.shape[0], radius, basis, sigmas


class WaveletNode(object):
    def __init__(self,
                 idxs: np.ndarray,
                 X: np.ndarray,
                 manifold_dim: int,
                 max_dim: int,
                 is_leaf: bool,
                 shelf = None,
                 threshold: float = 0.5,
                 precision: float = 1e-2) -> None:
        self.idxs: np.ndarray = idxs
        self.is_leaf: bool = is_leaf
        self.children: List[WaveletNodeType] = list()
        self.parent: WaveletNodeType = None

        self.center, self.size, self.radius, self.basis, self.sigmas = node_function(
            np.atleast_2d(X[idxs,:]),
            manifold_dim,
            max_dim,
            is_leaf,
            shelf=shelf,
            threshold=threshold,
            precision=precision
        )

        self.wav_basis: np.ndarray = None
        self.wav_sigmas: np.ndarray = None
        self.wav_consts: np.ndarray = None

    def make_transform(self,
                       X: np.ndarray,
                       manifold_dim: int,
                       max_dim: int,
                       shelf = None,
                       threshold: float = 0.5,
                       precision: float = 1e-2) -> None:
        parent_basis: np.ndarray = self.basis

        print(self.idxs.shape, self.basis.shape)
        if np.prod(parent_basis.shape) > 0:
            wav_dims: np.ndarray = np.zeros(len(self.children))
            for i,c in enumerate(self.children):
                if np.prod(c.basis.shape) > 0:
                    Y: np.ndarray = c.basis-(c.basis.dot(parent_basis.T)).dot(parent_basis)
                    _, s, V = rand_pca(Y, min(min(Y.shape), max_dim))

                    wav_dims[i] = (s > threshold).sum()
                    if wav_dims[i] > 0:
                        self.wav_basis = V[:,:wav_dims[i]].T
                        self.wav_sigmas = s[:wav_dims[i]]

                    self.wav_consts = c.center - self.center
                    self.wav_consts = self.wav_consts -\
                        parent_basis.T.dot(parent_basis*self.wav_consts)


class WaveletTree(object):
    def __init__(self,
                 dyadic_tree,
                 X: np.ndarray,
                 manifold_dims: Union[int, np.ndarray],
                 max_dim: int,
                 shelf = None,
                 thresholds: Union[float, np.ndarray] = 0.5,
                 precisions: Union[float, np.ndarray] = 1e-2) -> None:

        if not isinstance(manifold_dims, np.ndarray):
            self.manifold_dims = np.ones(dyadic_tree.num_levels, dtype=int)*\
                                 int(manifold_dims)
        if not isinstance(thresholds, np.ndarray):
            self.thresholds = np.ones(dyadic_tree.num_levels, dtype=float) * thresholds
        if not isinstance(precisions, np.ndarray):
            self.precisions = np.ones(dyadic_tree.num_levels, dtype=float) * precisions
        self.max_dim: int = max_dim
        self.shelf = shelf

        self.num_levels: int = dyadic_tree.num_levels
        self.root: WaveletNode = None
        self.num_nodes: int = 0

        self.make_basis(dyadic_tree, X)

    def make_basis(self,
                   dyadic_tree,
                   X: np.ndarray) -> None:
        cell_root = dyadic_tree.root
        self.root = WaveletNode(cell_root.idxs,
                                X,
                                self.manifold_dims[0],
                                self.max_dim,
                                len(cell_root.children) == 0,
                                shelf=self.shelf,
                                threshold=self.thresholds[0],
                                precision=self.precisions[0])
        self.num_nodes += 1

        current_cells = [cell_root]
        current_nodes = [self.root]

        for level in range(1, dyadic_tree.num_levels):
            next_cells = list()
            next_nodes = list()

            for cell, node in zip(current_cells, current_nodes):
                for child_cell in cell.children:
                    new_node = WaveletNode(child_cell.idxs,
                                           X,
                                           self.manifold_dims[level],
                                           self.max_dim,
                                           len(child_cell.children) == 0,
                                           shelf=self.shelf,
                                           threshold=self.thresholds[level],
                                           precision=self.precisions[level])

                    next_cells.append(child_cell)
                    next_nodes.append(new_node)
                    node.children.append(new_node)
                    new_node.parent = node
                    self.num_nodes += 1

            current_cells = next_cells
            current_nodes = next_nodes

    def make_wavelets(self,
                        X: np.ndarray) -> None:
        nodes_at_layers = [[self.root]]
        current_layer = nodes_at_layers[0]
        for level in range(1, self.num_levels):
            next_layer = list()
            for node in current_layer:
                print("level %s: basis.shape: %s" % (level, node.basis.shape))
                for child in node.children:
                    next_layer.append(child)

            nodes_at_layers.append(next_layer)
            current_layer = next_layer

        for j in range(self.num_levels-1, 0, -1):
            # built transforms
            nodes = nodes_at_layers[j]
            print("layer %s" % j)
            for node in nodes:
                node.make_transform(X, self.manifold_dims[j], self.max_dim,
                                    self.shelf, self.thresholds[j], self.precisions[j])

