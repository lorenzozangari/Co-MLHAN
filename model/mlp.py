"Multilayer perceptron"

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_dim, hidden, classes, dropout):
        super(MLP, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(in_dim, hidden, bias=True))
        self.fc_layers.append(nn.Linear(hidden, classes, bias=True))
        self.activation = F.relu
        self.softmax = F.softmax
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:

            self.dropout = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        """Learnable parameters init"""

        gain = nn.init.calculate_gain('relu')
        for m in self.fc_layers:
            nn.init.xavier_normal_(m.weight, gain=gain)

    def forward(self, x):
        h = x
        for i in range(len(self.fc_layers)-1):
            h = self.activation(self.fc_layers[i](h))
            h = self.dropout(h)
        logits = self.softmax(self.fc_layers[-1](h),  dim=-1)
        return logits