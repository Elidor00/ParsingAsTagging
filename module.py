from torch import nn
from numpy import prod


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

    def __init__(self, in_features, out_features, depth, activation='ReLU', dropout=0.3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        act_fn = getattr(nn, activation)
        for i in range(depth):
            self.layers.add_module('fc_{}'.format(i),
                                   nn.Linear(in_features, out_features))
            if activation:
                self.layers.add_module('{}_{}'.format(activation, i),
                                       act_fn())
            if dropout:
                self.layers.add_module('dropout_{}'.format(i),
                                       nn.Dropout(dropout))
            in_features = out_features

    def forward(self, x):
        return self.layers(x)

    @property
    def num_parameters(self):
        """Returns the number of trainable parameters of the model."""
        return sum(prod(p.shape) for p in self.parameters() if p.requires_grad)
