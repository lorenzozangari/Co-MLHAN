import argparse


def set_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save_emb', action="store_true")
    parser.add_argument('-d', '--dataset', type=str, default="imdb_mlh")
    parser.add_argument('--sa', action='store_true', help='True for CO-HAML SA')
    parser.add_argument('--node_lf', action="store_true")
    parser.add_argument('--target', type=str, default='M')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--metapath', type=str, default='MAM;MDM', help='Sequence of meta-paths separated by comma')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=72)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_hidden', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10000)

    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2_coef', type=float, default=0)

    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--feat_drop', type=float, default=0.3)
    parser.add_argument('--attn_drop', type=float, default=0.3)
    parser.add_argument('--lam', type=float, default=0.5)

    parser.add_argument('--checkpoint', type=str, default="checkpoint")
    parser.add_argument('--n_heads', type=int, default=1, help='Number of attentions heads (Q)')
    parser.add_argument('--no_replace',  action='store_true', help = 'sampling without'
                                                                     'replacement for Schema View')

    parser.add_argument('--rates', type=str, default='MA,7;MD,2')
    parser.add_argument('--features', type=str, default='M,features1000;A,identity;D,identity')

    parser.add_argument('--ncl', action='store_true', help='No cross-layer information for CO-HAML' )
    args, _ = parser.parse_known_args()
    return args
