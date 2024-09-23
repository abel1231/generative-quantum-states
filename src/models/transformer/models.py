import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer.utils import clones
from src.models.transformer.modules import LayerNorm


class Generator(nn.Module):
    """ linear + softmax layer for generation step """
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Decoder(nn.Module):
    """ this is the core of the transformer which is a stack n encoder layers """
    def __init__(self, layer, n_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """ pass the (masked) input trough all layers """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
