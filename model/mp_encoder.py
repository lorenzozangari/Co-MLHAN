import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(Attention, self).__init__()
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
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp


class MpEncoderSA(nn.Module):

    def __init__(self, in_dim, hidden_dim, p, attn_drop, n, num_hidden=1):
        super(MpEncoderSA, self).__init__()
        self.p = p
        self.num_hidden = num_hidden
        modules = {}
        for k in p:
            layers = nn.ModuleList()
            layers.append(dglnn.GraphConv(in_dim, hidden_dim, norm='both', activation=nn.PReLU(),
                          bias=True))
            for i in range(num_hidden - 1):
                layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, norm='both', activation=nn.PReLU(),
                              bias=True))
            modules[k] = layers

        self.node_level = nn.ModuleDict({k: modules[k] for k in p})
        self.att = Attention(hidden_dim, attn_drop)
        self.att_cl = Attention(hidden_dim, .0)
        self.n = n

    def forward(self, mps, target_feat):
        embeds = []
        for k in self.p:
            h = target_feat
            for i in range(self.num_hidden):
                layer = self.node_level[k][i]
                h = layer(mps[k], h)
            embeds.append(h)
        z_mp = self.att(embeds)
        z_mp = z_mp.view(-1, self.n, z_mp.shape[1])
        z_mp = [z_mp[i, ...] for i in range(z_mp.shape[0])]
        z_mp = self.att_cl(z_mp)
        return z_mp

    def inference(self, mps, target_feat):
        embeds = []
        for k in self.p:
            h = target_feat
            for i in range(self.num_hidden):
                layer = self.node_level[k][i]
                h = layer(mps[k], h)
            embeds.append(h)
        z_mp = self.att(embeds)
        z_mp = z_mp.view(-1, self.n, z_mp.shape[1])
        z_mp = [z_mp[i, ...] for i in range(z_mp.shape[0])]
        z_mp = self.att_cl(z_mp)
        return z_mp


class MpEncoder(nn.Module):

    def __init__(self, in_dim, hidden_dim, p, attn_drop, n, num_hidden=1):
        super(MpEncoder, self).__init__()
        self.p = p
        self.num_hidden = num_hidden
        modules = {}
        for k in p:
            layers = nn.ModuleList()
            layers.append(dglnn.GraphConv(in_dim, hidden_dim, norm='both', activation=nn.PReLU(),
                                          bias=True))
            for i in range(num_hidden - 1):
                layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, norm='both', activation=nn.PReLU(),
                                              bias=True))
            modules[k] = layers
        self.node_level = nn.ModuleDict({k: modules[k] for k in p})
        self.att = Attention(hidden_dim, attn_drop)
        self.n = n

    def forward(self, mps, target_feat):
        embeds = []

        for k in self.p:
            if isinstance(target_feat, dict):
                h = target_feat[k]
            else:
                h = target_feat

            for i in range(self.num_hidden):
                layer = self.node_level[k][i]
                h = layer(mps[k], h)
            embeds.append(h)
        z_mp = self.att(embeds)
        return z_mp

    def inference(self, mps, target_feat):
        embeds = []

        for k in self.p:
            h = target_feat
            for i in range(self.num_hidden):
                layer = self.node_level[k][i]
                h = layer(mps[k], h)
            embeds.append(h)
        z_mp = self.att(embeds)
        return z_mp
