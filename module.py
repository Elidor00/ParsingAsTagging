import numpy as np
import torch
import torch.nn.functional as F
from numpy import prod
from torch import nn


class MLP(nn.Module):
    """
    Module for an MLP with dropout.
    Args:
        in_features (~torch.Tensor):
            The size of each input feature.
        out_features (~torch.Tensor):
            The size of each output feature.
        depth (int):
            Depth of the MLP, often is 2, sometimes 1.
        activation:
            ReLU activation function.
        dropout (float):
            Default: 0.3.
    """

    def __init__(self, in_features, out_features, depth, flag, activation='ReLU', dropout=0.3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                   nn.Linear(in_features, out_features))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())
            if dropout and flag:
                self.layers.add_module('dropout_{}'.format(i),
                                       nn.Dropout(dropout))
            in_features = out_features

    def forward(self, x):
        return self.layers(x)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)


class SelfAttention(nn.Module):
    """
    Module for self-attention inspired by:
    https://gist.github.com/cbaziotis/94e53bdd6e4852756e0395560ff38aa4
    """

    def __init__(self, hidden_size, batch_first=True):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)

        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # matrix mult
        # apply attention layer
        # inputs: (batch_size, max_len, hidden_size)
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            .repeat(batch_size, 1, 1)  # (batch_size, hidden_size, 1)
                            )

        # batch size can be = 1 so, squeeze only in 2nd position
        attentions = torch.softmax(F.relu(weights.squeeze(2)), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).cuda()
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(inputs, attentions.unsqueeze(-1).expand_as(inputs))

        return weighted, attentions


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional, dropout):
        super(MyLSTM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             batch_first=batch_first,
                             bidirectional=bidirectional,
                             dropout=dropout)
        self.atten1 = SelfAttention(hidden_size * 2, batch_first=batch_first)  # 2 is bidrectional

        # self.lstm2 = nn.LSTM(input_size=hidden_size * 2,  # 140
        #                      hidden_size=hidden_size,  # 400
        #                      num_layers=num_layers,  # 2
        #                      batch_first=batch_first,  # True
        #                      bidirectional=bidirectional,  # True
        #                      dropout=dropout)
        # self.atten2 = SelfAttention(hidden_size * 2, batch_first=batch_first)

        # self.fc1 = nn.Sequential(nn.Linear(hidden_size * num_layers * 2, hidden_size * num_layers * 2),
        #                          nn.BatchNorm1d(hidden_size * num_layers * 2),
        #                          nn.ReLU())
        # self.fc2 = nn.Linear(hidden_size * num_layers * 2, 1) # 800
        # https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

    def forward(self, x, x_len):
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out1, (h_n, c_n) = self.lstm1(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x, att1 = self.atten1(x, lengths)  # skip connect

        # tmp1 = torch.bmm(x.unsqueeze(2), att1.unsqueeze(1))
        # tmpp1 = tmp1.transpose(1, 2)

        # out2, (h_n, c_n) = self.lstm2(out1)
        # y, lengths = nn.utils.rnn.pad_packed_sequence(out2, batch_first=True)
        # y, att2 = self.atten2(y, lengths)

        # tmp2 = torch.bmm(y.unsqueeze(2), att2.unsqueeze(1))
        # tmpp2 = tmp2.transpose(1, 2)

        # z = torch.cat([x, y], dim=2)
        # z = torch.cat([tmpp1, tmpp2], dim=2)
        # z = self.fc1(self.dropout(z))
        # z = self.fc2(self.dropout(z))
        return x
