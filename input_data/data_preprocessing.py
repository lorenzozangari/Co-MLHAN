"""
Script for data pre-preprocessing in CO-MLHAN
"""

import os
import pandas as pd
import numpy as np
import argparse
import dgl
import scipy as sp
import torch
import shutil
import traceback
import math
from pathlib import Path

root = '../data/'
prep = 'prep_data'
NSE = 'edges.txt'
NODES = 'nodes.txt'


def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="imdb_mlh")
    parser.add_argument('--metapath', type=str, default='MAM;MDM', help='Sequence of meta-paths separated by comma')
    parser.add_argument('--target', type=str, default='M')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--pos_th', type=int, default=5)
    parser.add_argument('--pos_cond', type=str, default="3;1", help='Conditions over '
                                                                'meta-path for positives construction.')
    parser.add_argument('--pos_w', type=str, default='0.5;0.5')
    parser.add_argument('--features', type=str, default='features1000',
                        help='Name of the file containing the features.')
    parser.add_argument('--node_lf', action='store_true', help='Features at node level.')

    args, _ = parser.parse_known_args()
    return args


def build_dgl_graph(rows, cols, num_nodes, simple=True):
    g = dgl.graph((rows, cols), num_nodes=num_nodes)
    if simple:
        return dgl.to_simple(g)
    return g


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def build_metapath(dataset, mps, n_layers, n_target):
    """
    :param dataset:
    :param mps: List of meta-paths (e.g. ['MAM', 'MDM',..]
    :param n_layers: number of layers of the multulayer graph
    :param n_target:
    :return:
    """
    terminal1 = 1
    terminal2 = -1
    for mp_name in mps:
        mpg = pd.read_csv(os.path.join(root, dataset, mp_name + ".txt"), delimiter=' ', header=None)
        if mpg.shape[1] >= 5:
            terminal2 = terminal2 - 1
        for l in range(n_layers):
            mpl = mpg.loc[mpg[0] == l]
            col = mpl.columns
            mpl = mpl[[col[terminal1], col[terminal2]]]
            mpl = mpl.to_numpy(dtype=np.dtype(np.int32))
            g = build_dgl_graph(mpl[:, 0], mpl[:, 1], num_nodes=n_target)
            # assert check_symmetric(g.adjacency_matrix().to_dense())
            print(f'Saving meta-path {mp_name} from layer {l} in {root}/{dataset}/{prep}/{mp_name}_{l}.bin')
            print('Graph nodes and edges ', g.number_of_nodes(), g.number_of_edges())
            dgl.save_graphs(f'{root}/{dataset}/{prep}/{mp_name}_{l}.bin', g)


def build_cl_metapath(dataset, mps, n_layers, n_target):
    """

    :param dataset:
    :param mps: List of meta-paths (e.g. ['MAM', 'MDM',..]
    :param n_layers: number of layers of the multulayer graph
    :param n_target:
    :return:
    """
    terminal1 = 2
    terminal2 = -1
    for mp_name in mps:
        f = os.path.join(root, dataset, mp_name + "_across.txt")
        if not os.path.isfile(f):
            print(f, ' does not exist!')
            continue
        mpg = pd.read_csv(f, delimiter=' ', header=None)
        for l1 in range(n_layers):
            for l2 in range(l1 + 1, n_layers):
                mpl = mpg.loc[(mpg[0] == l1) & (mpg[1] == l2)]
                if mpl.size == 0:
                    continue

                col = mpl.columns
                mpl = mpl[[col[terminal1], col[terminal2]]]
                mpl = mpl.to_numpy(dtype=np.dtype(np.int32))
                g = build_dgl_graph(mpl[:, 0], mpl[:, 1], num_nodes=n_target)
                # assert check_symmetric(g.adjacency_matrix().to_dense())
                print(
                    f'Saving meta-path {mp_name} from layers {l1}, {l2} in {root}/{dataset}/{prep}/{mp_name}_across_{l1}_{l2}.bin')
                print('Graph nodes and edges ', g.number_of_nodes(), g.number_of_edges())
                dgl.save_graphs(f'{root}/{dataset}/{prep}/{mp_name}_{l1}{l2}.bin', g)


def build_ml_metapath(dataset, mps, n_layers, n_target):
    dim = n_target * n_layers
    mp_graph = {}
    terminal1 = 1
    terminal2 = -1
    for mp_name in mps:
        print(f'Building supra adjacency matrix for meta-path {mp_name}')
        mpg = pd.read_csv(os.path.join(root, dataset, mp_name + ".txt"), delimiter=' ', header=None)
        rows = None
        cols = None
        for layer in range(n_layers):
            mpl = mpg.loc[mpg[0] == layer]
            col = mpl.columns
            mpl = mpl[[col[terminal1], col[terminal2]]]
            mpl = mpl.to_numpy(dtype=np.dtype(np.int32))
            src = mpl[:, 0]
            dst = mpl[:, 1]
            if layer == 0:
                rows = src
                cols = dst
            else:
                lrow = src + n_target * layer
                lcol = dst + n_target * layer
                rows = np.concatenate((rows, lrow))
                cols = np.concatenate((cols, lcol))

        adj = sp.sparse.csr_matrix((np.ones(rows.shape[0]), (rows, cols)), shape=(dim, dim))
        empty = np.ones(dim, dtype=bool)
        empty[np.unique(adj.nonzero())] = False
        data = 1
        # number of diagon als = (n_el -1) * 2
        offset = []
        print('Adding inter-layer edges...')
        for i in range(1, n_layers):
            x = i * n_target
            mx = -i * n_target
            offset.append(x)
            offset.append(mx)

        inter_adj = sp.sparse.diags([data] * len(offset), offset, shape=(dim, dim), format='csr')
        adj = adj + inter_adj
        g = dgl.from_scipy(adj)
        print(f'Saving meta-path {mp_name} in {root}/{dataset}/{prep}/{mp_name}.bin')
        dgl.save_graphs(f'{root}/{dataset}/{prep}/{mp_name}.bin', g)
        mp_graph[mp_name] = g
    return mp_graph


def load_target_entities(target):
    """
    :param target: str, indicating target type, e.g., 'M'.
    :return: Target entities
    """
    df = pd.read_csv(os.path.join(root, dataset, NODES), delimiter=' ', header=None).to_numpy()
    nodes = df[df[:, 1] == target][:, 2].astype(np.int32)
    return np.unique(nodes)


def load_number_of_entities(data, dataset):
    """
    :return: dictionary containing the number of entities for each type
    """
    df = pd.read_csv(os.path.join(data, dataset, NODES), delimiter=' ', header=None).to_numpy()
    types = np.unique(df[:, 1])
    d = {}
    for t in types:
        nodes = df[df[:, 1] == t][:, 2].astype(np.int32)
        d[t] = len(np.unique(nodes))
    return d


def save_nodes(dataset):
    src = os.path.join(root, dataset, NODES)
    dst = os.path.join(root, dataset, prep, NODES)
    shutil.copy(src, dst)


def build_network_schema(dataset, n_layers, target, num_nodes_dict):
    ns = pd.read_csv(os.path.join(root, dataset, NSE), delimiter=' ', header=None)
    relations = np.unique(ns[1].values)
    graph_data = {}
    for l in range(n_layers):
        for r in relations:
            if target not in r[:2]:
                continue
            rl_edges = ns.loc[(ns[0] == l) & (ns[1] == r)][[2, 3]].to_numpy(dtype=np.int32)
            if rl_edges.size == 0:
                raise ValueError(f'Relation {r} is missing for layer {l}')
            if r[0] == target:
                td = r[0]
                ts = r[1]
                src = rl_edges[:, 1]
                dst = rl_edges[:, 0]
            else:
                td = r[1]
                ts = r[0]
                src = rl_edges[:, 0]
                dst = rl_edges[:, 1]
            graph_data[(f'{ts}{l}', f'{r}{l}', f'{td}{l}')] = (torch.IntTensor(src), torch.IntTensor(dst))

    g = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)
    dgl.save_graphs(f'{root}/{dataset}/{prep}/intra_layer.bin', g)
    return g


def build_pos(dataset, n_target, pos_num, threeshold, weights):
    with open(os.path.join(root, dataset, "positives.txt")) as f:
        metapaths = f.readline()

    print('Metapaths : ', metapaths)
    all = np.zeros((n_target, n_target))
    mp = pd.read_csv(os.path.join(root, dataset, "positives.txt"), delimiter=' ', skiprows=1, header=None).values
    for row in mp:
        src = int(row[0])
        dst = int(row[1])
        cnt = .0
        for i, s in enumerate(threeshold):
            a = int(row[2 + i])
            if a >= s:
                cnt += weights[i]*a

        all[src, dst] = cnt

    pos = np.zeros((n_target, n_target))
    pos_num = pos_num - 1  # an entity always include itself as positive
    for i in range(len(all)):
        one = all[i].nonzero()[0]
        if len(one) > pos_num:
            oo = np.argsort(-all[i, one])
            sele = one[oo[:pos_num]]
            pos[i, sele] = 1
        else:
            pos[i, one] = 1
        pos[i, i] = 1

    pos = sp.sparse.coo_matrix(pos)
    sp.sparse.save_npz(f'{root}/{dataset}/{prep}/pos.npz', pos)
    return pos


def build_features(dataset, feats_name, tps, n_layers, EL=True):
    f = pd.read_csv(os.path.join(root, dataset, feats_name + ".txt"), delimiter=' ', header=None).fillna(.0)
    if EL:
        for t in tps:
            ft = f.loc[f[0] == t]
            if not ft.empty:
                ft.drop(0, axis=1, inplace=True)
                ft = ft.values
                ft = ft[ft[:, 0].argsort()]
                ft = ft[:, 1:]
                feats = sp.sparse.coo_matrix(ft)
                print('Saving features ', os.path.join(root, dataset, prep, f"{t}_{feats_name}.npz"))
                sp.sparse.save_npz(os.path.join(root, dataset, prep, f"{t}_{feats_name}.npz"), feats)
    else:

        for t in tps:
            for l in range(n_layers):
                ft = f.loc[(f[0] == l) & (f[1] == t)]
                if not ft.empty:
                    ft = ft.drop([0, 1], axis=1)
                    ft = ft.values
                    ft = ft[ft[:, 0].argsort()]
                    ft = ft[:, 1:]
                    feats = sp.sparse.coo_matrix(ft)
                    print('Saving features', os.path.join(root, dataset, prep, f"{t}_{l}_{feats_name}.npz"))
                    sp.sparse.save_npz(os.path.join(root, dataset, prep, f"{t}_{l}_{feats_name}.npz"), feats)


if __name__ == '__main__':

    args = set_params()
    mps = list(map(str.upper, args.metapath.split(';')))
    dataset = args.dataset
    n_layers = args.layers
    target = args.target.upper()
    pos_th = args.pos_th
    pos_cond = list(map(int, args.pos_cond.split(';')))
    EL = not args.node_lf
    feats_name = args.features
    weights = list(map(float, args.pos_w.split(';')))

    assert math.isclose(np.sum(weights), 1.0)

    Path(os.path.join(root, dataset, prep)).mkdir(parents=True, exist_ok=True)

    print(f'Dataset : {dataset} \nMeta-path(s) : {mps} \nNumber of layers : {n_layers} \n'
          f'target entity : {target} \npos_th = {pos_th} \npos_cond = {pos_cond} \n'
          f'Weights pos = {weights} \nfeatures = {feats_name} \nEntity-level = {EL} ')

    print()
    print('Saving nodes...')
    try:
        save_nodes(dataset)
    except Exception as e:
        print('Error in saving nodes')
        print(e)
    print('Loading target entities....')
    flag = False
    entities = None
    try:
        entities = load_target_entities(target)
    except Exception as e:
        print('Error in saving nodes')
        print('Cannot construct meta-paths and positives because of missing entity information!')

        print(e)
        flag = not flag

    if not flag:
        n_target = len(entities)
        print('Number of target entities :', n_target)
        try:
            print('Building meta-path for each layer...')
            build_metapath(dataset, mps, n_layers, n_target)
            print('Building across-layer meta-path...')
            build_cl_metapath(dataset, mps, n_layers, n_target)
            print('Building meta-path based graph....')
            build_ml_metapath(dataset, mps, n_layers, n_target)
        except Exception as e:
            print('Error in meta-path construction!')
            print(e)
            traceback.print_exc()
            ### Positives

        try:
            build_pos(dataset, n_target, pos_th, pos_cond, weights)
        except Exception as e:
            print('Error in positive building')
            traceback.print_exc()
        print('Positives construction done!')

    print('Building network schema graph....')
    flag = False
    nd = None
    tps = None
    try:
        nd = load_number_of_entities(root, dataset)
        tps = list(nd.keys())
        print('Number of entities:\n', nd)
        print('Types of nodes:\n', tps)
    except Exception as e:
        print('Error in loading entities information!')
        print('Cannot construct network schema because of missing entity information!')
        print(e)
        flag = not flag
    if not flag:
        num_nodes_dict = {}
        for tp in nd:
            for l in range(n_layers):
                num_nodes_dict[tp + str(l)] = nd[tp]

        g = build_network_schema(dataset, n_layers, target, num_nodes_dict)
        for r in g.canonical_etypes:
            print(f'Number of edges for relation {r} : {g.number_of_edges(r)} ')

        print('Graph construction done!')

    print('Features building...')
    try:
        build_features(dataset, feats_name=feats_name, tps=tps, n_layers=n_layers, EL=EL)
    except Exception as e:
        print('Error in features building! ')
        print(e)

    print('Finished')
