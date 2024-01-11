import torch.nn as nn
import torch.nn.functional as F
from model.mp_encoder import MpEncoderSA, MpEncoder
from model.sc_encoder import ScEncoder
import torch


class ComlhanSa(nn.Module):

    def __init__(self, hidden_dim, feats_dim_dict, feat_drop, attn_drop, p, target_node,
                 rel_names, nl, n, num_heads, num_hidden, node_lf):

        super(ComlhanSa, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_dict = nn.ModuleDict({k: nn.Linear(feats_dim, hidden_dim, bias=True) for
                                      k, feats_dim in feats_dim_dict.items()})

        self.node_lf = node_lf

        for fc in self.fc_dict.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.target_node = target_node
        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.mp = MpEncoderSA(hidden_dim, hidden_dim, p, attn_drop, n, num_hidden)
        self.sc = ScEncoder(hidden_dim, hidden_dim, attn_drop, rel_names,
                            target_node, nl, num_heads, node_lf)

        self.nl = nl
        self.num_hidden = num_hidden

    def forward(self, mps, gns_loader, feats):
        h_all = {}
        for k in feats:
            h_all[k] = F.elu(self.feat_drop(self.fc_dict[k](feats[k])))

        if self.node_lf:
            ht = self.build_nlf(h_all)
        else:
            ht = torch.cat([h_all[self.target_node]] * self.nl, dim=0)

        z_mp = self.mp(mps, ht)
        z_sc = self.sc(gns_loader, h_all)

        return z_mp, z_sc

    def get_sc_embeds(self, gns_loader, feats):
        h_all = {}
        for k in feats:
            h_all[k] = F.elu(self.fc_dict[k](feats[k]))

        z_sc = self.sc(gns_loader, h_all)
        return z_sc

    def inference(self, mps, gns, feats):
        h_all = {}
        for k in feats:
            h_all[k] = F.elu(self.fc_dict[k](feats[k]))

        ht = torch.cat([h_all[self.target_node]] * self.nl, dim=0)
        z_mp = self.mp.inference(mps, ht)
        z_sc = self.sc.inference(gns, h_all)
        return z_mp, z_sc

    def get_embeds(self, mps, feats):
        if self.node_lf:
            z_mp = {}
            for k in feats:
                if self.target_node in k:
                    z_mp[k] = F.elu(self.fc_dict[k](feats[k]))

            z_mp = self.build_nlf(z_mp)
        else:
            z_mp = F.elu(self.fc_dict[self.target_node](feats[self.target_node]))
            z_mp = torch.cat([z_mp] * self.nl, dim=0)

        z_mp = self.mp(mps, z_mp)
        return z_mp.detach()

    def build_nlf(self, h_all):
        ht = [None] * self.nl
        for k in h_all:
            if k[0] == self.target_node:
                i = int(k[1])
                ht[i - 1] = h_all[k]
        ht = torch.cat(ht, dim=0)
        return ht


class Comlhan(nn.Module):

    def __init__(self, hidden_dim, feats_dim_dict, feat_drop, attn_drop, p, target_node,
                 rel_names, nl, n, num_heads, num_hidden, node_lf):
        """

        :param hidden_dim: int. hidden dimension
        :param feats_dim_dict: dict. input dimension
        :param feat_drop: features dropout
        :param attn_drop: attention dropout
        :param p: dict. meta-path name
        :param target_node: target node type
        :param rel_names: relation names
        :param nl: number of layers
        :param n: number of entities
        :param num_heads: attention heads
        """
        super(Comlhan, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_dict = nn.ModuleDict({k: nn.Linear(feats_dim, hidden_dim, bias=True) for
                                      k, feats_dim in feats_dim_dict.items()})

        for fc in self.fc_dict.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        if feat_drop > 0:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x

        self.mp = MpEncoder(hidden_dim, hidden_dim, p, attn_drop, n, num_hidden)
        self.sc = ScEncoder(hidden_dim, hidden_dim, attn_drop, rel_names, target_node, nl,
                            num_heads, node_lf)
        self.nl = nl
        self.target_node = target_node
        self.num_hidden = num_hidden
        self.node_lf = node_lf

    def forward(self, mps, gns_loader, feats):

        h_all = {}
        for k in feats:
            h_all[k] = F.elu(self.feat_drop(self.fc_dict[k](feats[k])))
        if self.node_lf:
            ht = self.build_nlf(mps, h_all)
        else:
            ht = h_all[self.target_node]

        z_mp = self.mp(mps, ht)
        z_sc = self.sc(gns_loader, h_all)
        return z_mp, z_sc

    def inference(self, mps, gns, feats):
        h_all = {}
        for k in feats:
            h_all[k] = F.elu(self.feat_drop(self.fc_dict[k](feats[k])))

        z_mp = self.mp.inference(mps, h_all[self.target_node])
        z_sc = self.sc.inference(gns, h_all)
        return z_mp, z_sc

    def get_embeds(self, mps, feats):
        if self.node_lf:
            z_mp = {}
            for k in feats:
                if self.target_node in k:
                    z_mp[k] = F.elu(self.fc_dict[k](feats[k]))
            z_mp = self.build_nlf(mps, z_mp)
        else:
            z_mp = F.elu(self.fc_dict[self.target_node](feats[self.target_node]))
        z_mp = self.mp(mps, z_mp)
        return z_mp.detach()

    def get_sc_embeds(self, gns_loader, feats):
        h_all = {}
        for k in feats:
            h_all[k] = F.elu(self.fc_dict[k](feats[k]))

        z_sc = self.sc(gns_loader, h_all)
        return z_sc

    def build_nlf(self, mps, h_all):
        ht = {}
        for mp in mps:
            index = mp.rfind('_')
            layer = mp[index+1:]
            if len(layer) == 2:
                src_layer = layer[0]
                dst_layer = layer[1]
                feats_src = h_all[self.target_node+src_layer]
                feats_dst = h_all[self.target_node+dst_layer]
                feat_bots = (feats_src + feats_dst)/2
                src_nodes = mps[mp].ndata['src_nodes']
                dst_nodes = mps[mp].ndata['dst_nodes']
                feats = feat_bots
                feats[src_nodes] = feats_src[src_nodes]
                feats[dst_nodes] = feats_dst[dst_nodes]
                ht[mp] = feats
            elif len(layer) == 1:
                src_layer = layer[0]
                feats = h_all[self.target_node+src_layer]
                ht[mp] = feats
            else:
                raise ValueError('Too many layers')
        return ht
