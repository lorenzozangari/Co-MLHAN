import numpy as np
import torch
import scipy.sparse as sp


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def gaussian_distribution(num_features, n):
    return torch.from_numpy(np.random.normal(size=(num_features, n)).T).to(torch.float32)


def exponential_distribution(num_features, n):
    return torch.from_numpy(np.random.exponential(size=(num_features, n)).T).to(torch.float32)


def uniform_distribution(num_features, n):
    return torch.from_numpy(np.random.uniform(size=(num_features, n)).T).to(torch.float32)


def mixed_distribution(num_features, n):
    g = int(num_features / 3)
    x1 = np.random.normal(size=(g, n)).T
    x2 = np.random.exponential(size=(g, n)).T
    x3 = np.random.uniform(size=(num_features - g * 2, n)).T
    return torch.from_numpy(np.hstack((x1, x2, x3))).to(torch.float32)


def identity_matrix(n, sparse=False):
    if sparse:
        values = torch.ones(n, dtype=torch.float32)
        indices = np.vstack((torch.arange(n), torch.arange(n)))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = (n, n)
        identity = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        return identity

    return torch.eye(n)



def preprocess_features(features, to_dense=True):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if to_dense:
        return features.todense()
    return features


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())