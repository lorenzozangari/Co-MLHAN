import numpy as np
import scipy.sparse as sp
from util.util import sparse_mx_to_torch_sparse_tensor
import os


def get_random_pos(dim, probs):
    """Generate random positives"""
    pos = np.random.choice([0, 1], size=dim*dim, p=probs).reshape(dim, dim)
    pos = sp.coo_matrix(pos)
    return sparse_mx_to_torch_sparse_tensor(pos)


def identity_pos(dim):
    return sparse_mx_to_torch_sparse_tensor(sp.identity(dim))


def load_pos(path):
    pos = sp.load_npz(os.path.join(path, "pos.npz"))
    pos = sparse_mx_to_torch_sparse_tensor(pos)
    return pos
