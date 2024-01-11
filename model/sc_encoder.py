import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
from model.gatv2 import GATv2Conv


class InterAtt(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(InterAtt, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        z_mc = 0
        for i in range(len(embeds)):
            z_mc += embeds[i] * beta[i]
        return z_mc


def extract_embed(node_embed, input_nodes, node_lf=False):
    emb = {}
    if node_lf:
        for ntype, nid in input_nodes.items():
            nid = input_nodes[ntype].long()
            emb[ntype] = node_embed[ntype][nid]
    else:
        for ntype, nid in input_nodes.items():
            nid = input_nodes[ntype].long()
            emb[ntype] = node_embed[ntype[0]][nid]
    return emb


class ScEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, attn_drop, rel_names, target_node, nl,
                 num_heads=1, node_lf=False):

        super(ScEncoder, self).__init__()

        self.intra = dglnn.HeteroGraphConv({
            rel: GATv2Conv(in_dim, hidden_dim, num_heads, attn_drop=attn_drop,
                           bias=True, activation=F.elu)
            for rel in rel_names}, aggregate=lambda tensors, _: tensors)

        self.inter = nn.ModuleList([InterAtt(hidden_dim, attn_drop=attn_drop)
                                    for _ in range(nl)])

        self.cross_attn = InterAtt(hidden_dim, .0)
        self.target_node = target_node
        self.nl = nl
        self.rel_names = rel_names
        self.num_heads = num_heads
        self.node_lf = node_lf

    def forward(self, node_loader, feats):
        h = feats
        for i, (input_nodes, output_nodes, blocks) in enumerate(node_loader):
            h = extract_embed(feats, input_nodes, self.node_lf)
            h = self.intra(blocks[0], h)

        for k, v in h.items():
            h[k] = [x.mean(1) for x in v]
        l_emb = []
        for i, inter in enumerate(self.inter):
            target = f'{self.target_node}{i}'
            l_emb.append(inter(h[target]))

        return self.cross_attn(l_emb)

    def inference(self, gns, feats):
        h = build_sc_feats(gns, feats)
        h = self.intra(gns, h)
        for k, v in h.items():
            h[k] = [x.mean(1) for x in v]
        l_emb = []

        for i, inter in enumerate(self.inter):
            target = f'{self.target_node}{i}'
            l_emb.append(inter(h[target]))

        return self.cross_attn(l_emb)


def build_sc_feats(gns, feats_dic):
    h_ml = {}
    for k in gns.ntypes:
        e = k[0]
        h_ml[k] = feats_dic[e]
    return h_ml



