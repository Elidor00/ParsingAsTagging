import torch
from numpy import prod
from torch import nn
from torch.nn import functional as F, BatchNorm1d
from torch.autograd import Variable
from torch.nn.modules.activation import MultiheadAttention


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

    def __init__(self, in_features, out_features, depth, flag, before_act, after_act,
                 activation='ReLU', dropout=0.3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                   nn.Linear(in_features, out_features))

            if before_act:
                self.layers.add_module('bn_{}'.format(i),
                                       BatchNorm1d(out_features))

            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())

            if after_act:
                self.layers.add_module('bn_{}'.format(i),
                                       BatchNorm1d(out_features))

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


class Norm(nn.Module):
    """
    Normalising results
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class SelfAttention(nn.Module):
    """
    Self-Attention module
    """
    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = nn.Parameter(torch.FloatTensor(attention_size), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        nn.init.uniform_(self.attention_weights.data, -0.005, 0.005)

    @staticmethod
    def get_mask(attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        scores = scores.unsqueeze(-1).expand_as(inputs)
        weighted = torch.mul(inputs, scores)
        # sum the hidden states
        # representations = weighted.sum(1).squeeze()
        representations = weighted

        return representations, scores


class MyLSTMSA(nn.Module):
    """
    LSTM with Self-Attention and Batch Normalization
    """
    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional, dropout):
        super(MyLSTMSA, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.dropout = nn.Dropout(p=dropout)
        self.lstm1 = nn.LSTM(input_size=self.input_size,  # 140
                             hidden_size=self.hidden_size,  # 400
                             num_layers=self.num_layers,  # 2
                             batch_first=self.batch_first,  # True
                             bidirectional=self.bidirectional,  # True
                             dropout=dropout)
        self.batch_norm = BatchNorm1d(self.hidden_size * 2, affine=True)
        # self.norm_1 = Norm(self.hidden_size * 2)
        self.atten1 = SelfAttention(self.hidden_size * 2, batch_first=batch_first)  # 2 is bidirectional

    @staticmethod
    def simple_elementwise_apply(fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def forward(self, x, x_len):
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out1, (h_n, c_n) = self.lstm1(x)
        # out1 = (batch, seq_len, num_directions * hidden_size)
        # h_n = (num_layers * num_directions, batch, hidden_size)
        out1 = self.simple_elementwise_apply(self.batch_norm, out1)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x, att1 = self.atten1(x, lengths)  # skip connect
        return x


class MyLSTM_MHSA(nn.Module):
    """
    LSTM with Multi Head (Self) Attention
    """
    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional, dropout):
        super(MyLSTM_MHSA, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.dropout = nn.Dropout(p=dropout)
        self.lstm1 = nn.LSTM(input_size=input_size,  # 140
                             hidden_size=hidden_size,  # 400
                             num_layers=num_layers,  # 3
                             batch_first=batch_first,  # True
                             bidirectional=bidirectional,  # True
                             dropout=dropout)
        self.norm_1 = Norm(self.hidden_size * 2)
        self.multi_att = MultiheadAttention(self.hidden_size * 2, 10)  # * 2 = bidirectional

    @staticmethod
    def simple_elementwise_apply(fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def forward(self, x, x_len):
        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out1, (h_n, c_n) = self.lstm1(x)
        # out1 = (seq_len, batch, num_directions * hidden_size)
        # h_n = (num_layers * num_directions, batch, hidden_size)
        out1 = self.simple_elementwise_apply(self.norm_1, out1)
        x, lengths = nn.utils.rnn.pad_packed_sequence(out1, batch_first=True)
        x_new = torch.transpose(x, 1, 0)
        x, att1 = self.multi_att(x_new, x_new, x_new)
        x_out = torch.transpose(x, 0, 1)
        return x_out


class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size):  # 400,400,len(deprel)
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)  # 401,401,len(deprel)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class DeepBiaffineScorer(nn.Module):
    """
    Biaffine Scorer inspired by Dozat and Manning
    https://arxiv.org/abs/1611.01734
    """
    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0):
        # input1_size = 800 (400*2), input2_size = 800 (400*2), hidden_size = 400, output_size = X
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)  # 800, 400
        self.W2 = nn.Linear(input2_size, hidden_size)  # 800, 400
        self.hidden_func = hidden_func
        self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)  # 400,400,len(deprel)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))),
                           self.dropout(self.hidden_func(self.W2(input2))))
