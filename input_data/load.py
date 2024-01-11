import os
import dgl
import numpy as np
import torch
from input_data.pos import load_pos, identity_pos, get_random_pos
from util.util import gaussian_distribution, \
    uniform_distribution, mixed_distribution, exponential_distribution, identity_matrix, \
    preprocess_features
import pandas as pd
from input_data.data_preprocessing import load_number_of_entities
import scipy.sparse as sp
root = './data/'
prep = 'prep_data/'

_k_distr = ['normal', 'identity', 'uniform', 'mixed', 'exponential']


def _load_data(dataset, target, metapath, n_layers, n_e, ncl=False, sa=False, num_features=64,
               add_self_loop=True, f_distribution=None, node_level_fs=False):

    path = os.path.join(root, dataset, prep)

    # Load network schema graph
    intra_layer_path = os.path.join(path, 'intra_layer.bin')
    gns = dgl.load_graphs(intra_layer_path)[0][0]

    # Load meta-path graphs
    mps = {}
    if sa:  # if sa model

        for mp in metapath:
            mp = mp.upper()
            mps[mp] = dgl.load_graphs(os.path.join(path, f'{mp}.bin'))[0][0]
    else:

        for l in range(n_layers):
            for mp in metapath:
                mps[f'{mp}_{l}'] = dgl.load_graphs(os.path.join(path, f'{mp}_{l}.bin'))[0][0]

        if not ncl:
            for mp in metapath:
                for l1 in range(n_layers):
                    for l2 in range(l1+1, n_layers):
                        f = os.path.join(path, f'{mp}_{l1}{l2}.bin')
                        if not os.path.isfile(f):
                            print(f, ' does not exist!')
                            continue
                        mps[f'{mp}_{l1}{l2}'] = dgl.load_graphs(f)[0][0]

        if node_level_fs:
            for k in mps:
                index = k.rfind('_')
                layer = k[index+1:]
                if len(layer) == 2:
                    src_nodes = mps[k].edges()[0].unique().numpy()
                    dst_nodes = mps[k].edges()[1].unique().numpy()
                    both_nodes = np.intersect1d(src_nodes, dst_nodes)
                    src_nodes = np.setdiff1d(src_nodes, both_nodes)
                    dst_nodes = np.setdiff1d(dst_nodes, both_nodes)
                    src_m = torch.zeros(n_e[target], dtype=torch.bool)
                    dst_m = torch.zeros(n_e[target], dtype=torch.bool)
                    both_m = torch.zeros(n_e[target], dtype=torch.bool)
                    src_m[src_nodes] = True
                    dst_m[dst_nodes] = True
                    both_m[both_nodes] = True
                    mps[k].ndata['src_nodes'] = src_m
                    mps[k].ndata['dst_nodes'] = dst_m
                    mps[k].ndata['both_nodes'] = both_m
                elif len(layer) > 2:
                    raise ValueError('Unsupported number of layers! Number of layers could be greater than 10.')

    if add_self_loop:
        for k in mps:
            mps[k] = dgl.add_self_loop(mps[k])

    feats = {}
    if f_distribution is None:
        for k in n_e:
            if not node_level_fs:
                feats[k] = torch.from_numpy(np.random.normal(size=(num_features, n_e[k])).T).to(torch.float32)
            else:
                for l in range(n_layers):
                    feats[k+str(l)] = torch.from_numpy(np.random.normal(size=(num_features, n_e[k])).T).to(torch.float32)
    else:
        for k, d in f_distribution.items():
            if d == 'normal':
                if not node_level_fs:
                    feats[k] = gaussian_distribution(num_features, n_e[k])
                else:
                    feats[k] = gaussian_distribution(num_features, n_e[k[0]])
            elif d == 'uniform':
                if not node_level_fs:
                    feats[k] = uniform_distribution(num_features, n_e[k])
                else:
                    feats[k] = uniform_distribution(num_features, n_e[k[0]])
            elif d == 'exponential':
                if not node_level_fs:
                    feats[k] = exponential_distribution(num_features, n_e[k])
                else:
                    feats[k] = exponential_distribution(num_features, n_e[k[0]])
            elif d == 'mixed':
                if not node_level_fs:
                    feats[k] = mixed_distribution(num_features, n_e[k])
                else:
                    feats[k] = mixed_distribution(num_features, n_e[k[0]])
            elif d == 'identity':
                if not node_level_fs:
                    feats[k] = identity_matrix(n_e[k], sparse=True)
                else:
                    feats[k] = identity_matrix(n_e[k[0]], sparse=True)
            else:
                print(f'Loading features of {k} from file {d+".npz"}...')
                if not node_level_fs:
                    feats[k] = torch.FloatTensor(
                        preprocess_features(sp.load_npz(os.path.join(path, d+".npz")))).to_sparse()

                else:
                    feats[k] = torch.FloatTensor(preprocess_features(
                            sp.load_npz(os.path.join(path, d+".npz")))).to_sparse()
    pos = load_pos(path)

    return gns, mps, feats, pos


def load_data(args):
    dataset = args.dataset
    target = args.target
    metapath = list(map(str.upper, args.metapath.split(';')))
    n_layers = args.layers
    n_e = load_number_of_entities(root, os.path.join(dataset, prep))

    f_distribution = {}
    input_features = args.features.split(';')

    if not args.node_lf:
        for fi in input_features:
            fir = fi.split(',')
            if fir[1] in _k_distr:
                ffn = fir[1]
            else:
                ffn = f'{fir[0].upper()}_{fir[1]}'

            f_distribution[fir[0].upper()] = ffn
    else:
        for fi in input_features:
            fir = fi.split(',')
            tp = fir[0].upper()
            ff = fir[1]
            for l in range(n_layers):
                if ff in _k_distr:
                    ffn = ff
                else:
                    ffn = f'{tp}_{l}_{ff}'
                f_distribution[tp+str(l)] = ffn

    gns, mps, feats, pos = _load_data(dataset, target, metapath, n_layers, n_e, args.ncl, args.sa,
                                      f_distribution=f_distribution, node_level_fs=args.node_lf)

    sample_rate = {}
    rates = args.rates
    r_list = rates.split(';')
    for rl in r_list:
        rt = rl.split(',')
        t = rt[0]
        rate = int(rt[1])

        for et in gns.canonical_etypes:
            if t in et[1]:
                sample_rate[et] = rate

    args.sample_rate = sample_rate
    print('Sample rate : ', args.sample_rate)

    return gns, mps, feats, pos, n_e



