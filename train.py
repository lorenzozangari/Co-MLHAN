"""Main module for training"""
from input_data.params import set_params
from input_data.load import load_data
import torch
import random
import numpy as np
import dgl
from util.early_stopping import EarlyStopping
from model.co_mlhan import ComlhanSa, Comlhan
from model.contrast import Contrast
import time
from pathlib import Path


save_dir = './artifacts/'


def build_model(args, feats_dim_dic, p, rel_names, nl, n):
    if args.sa:
        model = ComlhanSa(args.hidden_dim, feats_dim_dic, args.feat_drop, args.attn_drop,
                          p, args.target, rel_names, nl, n, args.n_heads, args.num_hidden,
                          args.node_lf)
    else:
        model = Comlhan(args.hidden_dim, feats_dim_dic, args.feat_drop, args.attn_drop,
                        p, args.target, rel_names, nl, n, args.n_heads, args.num_hidden, args.node_lf)
    return model


def get_loss(hidden_dim, tau, lam):
    return Contrast(hidden_dim, tau, lam)


def save_embedding(args, embeds, node_lf, base_name=None, final=True):
    name = 'final' if final else 'initial'
    name = base_name + "_" + name if base_name is not None else name
    mdir = f'CO_HAML_seed_{args.seed}/' if not args.sa else f'CO_HAML SA_seed_{args.seed}/'
    lf_dir = 'node_lf/' if node_lf else "type_lf/"
    dr = save_dir + mdir + lf_dir
    Path(dr).mkdir(parents=True, exist_ok=True)
    torch.save(embeds.cpu(), dr + args.dataset + "_" + name + ".pt")


def save_model(args, model, node_lf):
    mdir =  f'CO_HAML_seed_{args.seed}/' if not args.sa else f'CO_HAML SA_seed_{args.seed}/'
    lf_dir = 'node_lf/' if node_lf else "type_lf/"
    dr = save_dir + mdir + lf_dir
    Path(dr).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dr + args.dataset + ".bin")


def build_data_loader(sample_rate, gns, target_node, device=torch.device('cpu'),
                      replace=True):
    sample_rate = [sample_rate]
    # Sampling with replacement
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_rate, replace=replace)
    train_nid_dict = {k: gns.nodes(k) for k in gns.ntypes if k[0] == target_node}
    num_nodes = sum([gns.num_nodes(k) for k in gns.ntypes if k[0] == target_node])
    dataloader = dgl.dataloading.DataLoader(
        gns, train_nid_dict, sampler,
        batch_size=num_nodes, device=device, shuffle=False, drop_last=False, num_workers=1)
    return dataloader


def train(args):
    cuda = torch.cuda.is_available() and not args.gpu < 0
    if cuda:
        torch.cuda.set_device(args.gpu)
        print('GPU device {}'.format(torch.cuda.get_device_name(args.gpu)))
        device = torch.device("cuda:" + str(args.gpu))
    else:
        device = torch.device("cpu")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    gns, mps, feats, pos, n_e = load_data(args)
    feats_dim_dic = {k: feats[k].shape[1] for k in feats}
    p = mps.keys()
    target_node = args.target
    n = n_e[target_node]

    print('Features size ', feats_dim_dic)
    print(f'Meta paths : {p} ; Target : {target_node}; Entities : {n_e} ')
    print('Sample rate : ', args.sample_rate)
    print()

    node_loader = build_data_loader(args.sample_rate, gns, target_node, device=device,
                                    replace=not args.no_replace)
    model = build_model(args, feats_dim_dic, p, gns.etypes, args.layers, n)
    loss = get_loss(args.hidden_dim, args.tau, args.lam)
    if cuda:
        model.cuda()
        feats = {k: feat.cuda() for k, feat in feats.items()}
        mps = {k: mp.to(device) for k, mp in mps.items()}
        gns = gns.to(device)
        pos = pos.cuda()
        loss.cuda()

    stopper = EarlyStopping(patience=args.patience, maximize=False,
                            model_name=args.dataset+"_" + str(seed),
                            model_dir=args.checkpoint)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_coef)

    epochs = args.epochs
    loss_list = []
    t_total = time.time()
    for epoch in range(epochs):
        model.train()  # training stage
        opt.zero_grad()
        z_mp, z_sc = model(mps, node_loader, feats)
        loss_value = loss(z_mp, z_sc, pos)
        loss_s = loss_value.data.cpu().item()
        loss_list.append(loss_s)
        print('Epoch {:05d} | Loss Val {:.4f}'.format(epoch + 1, loss_s))
        if stopper.step(loss_value, model, epoch):
            break
        loss_value.backward()
        opt.step()

    print("Optimization finished")
    print("Total training time: {:.4f}s".format(time.time() - t_total))

    print('Loading the best model, epoch {}....'.format(stopper.best_epoch+1))
    model.load_state_dict(torch.load(stopper.save_dir))
    model.eval()

    if args.save_emb:
        try:
            embeds = model.get_embeds(mps, feats)
            save_embedding(args, embeds, args.node_lf, final=True)
        except:
            print('Cannot save meta-path view embedding.')
        try:
            save_model(args, model, args.node_lf)
        except:
            print('Cannot save the model.')

    stopper.remove_checkpoint()


if __name__ == '__main__':

    def print_arguments(args):
        print()
        print('List of arguments: ')
        if isinstance(args, dict):
            for i, (k, v) in enumerate(args.items()):
                print("{} : {}".format(k, v), end="; ")
                if (i + 1) % 7 == 0:
                    print()

        else:
            for i, arg in enumerate(vars(args)):
                print("{} : {}".format(arg, getattr(args, arg)), end="; ")
                if (i + 1) % 7 == 0:
                    print()
        print()

    param = set_params()
    print_arguments(param)
    train(param)

