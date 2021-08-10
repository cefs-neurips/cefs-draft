import numpy as np
import torch
import torch.nn as nn

from .activation import ActNorm, Squeeze
from .base import Sequential, FlowComponent
from .conv import AffineCoupling, Invertible1x1Conv


class GlowStep(Sequential):
    '''
    A single step of a Glow network, depicted in figure 2 (a) of the paper:
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma and Prafulla Dhariwal
    https://arxiv.org/pdf/1807.03039.pdf
    '''

    invertible = True
    conformal = False

    def __init__(self, channels, additive_coupling=False, hidden_size=512):
        super().__init__(
            ActNorm(channels),
            Invertible1x1Conv(channels),
            AffineCoupling(channels, additive=additive_coupling, hidden_size=hidden_size),
        )


class GlowNet(FlowComponent):
    '''
    Based on "Glow: Generative Flow with Invertible 1x1 Convolutions"
    by Diederik P. Kingma and Prafulla Dhariwal
    https://arxiv.org/pdf/1807.03039.pdf
    '''

    invertible = True
    conformal = False

    def __init__(self, x_channels, k=32, l=3, hidden_size=512, out_shape=None,
                 additive_coupling=False):
        super().__init__()

        self.k = k
        self.l = l
        self.x_channels = x_channels
        self.levels = nn.ModuleList()
        self.out_shape = out_shape
        channels = x_channels

        # Build up each of L levels of the flow as per figure 2 of the paper
        for _ in range(l):
            channels *= 4 # The squeeze quadruples the channel count
            level = Sequential(
                Squeeze(),
                *[GlowStep(channels, additive_coupling=additive_coupling, hidden_size=hidden_size)
                  for _ in range(k)]
            )
            channels //= 2 # The split halves the channel count
            self.levels.append(level)

    def initialize(self, x):
        z = []

        for level in self.levels[:-1]:
            x = level.initialize(x)
            z_i, x = x.chunk(2, dim=1)
            if self.out_shape is not None:
                z.append(z_i.reshape((x.shape[0], -1) + self.out_shape))
            else:
                z.append(z_i.flatten(start_dim=1))

        if self.out_shape is not None:
            z_l = self.levels[-1].initialize(x).reshape((x.shape[0], -1) + self.out_shape)
        else:
            z_l = self.levels[-1].initialize(x).flatten(start_dim=1)
        z.append(z_l)

        return torch.cat(z, dim=1)

    def data_to_latent(self, x, m):
        log_det = 0.0
        z = []

        for level in self.levels[:-1]:
            x, level_log_det = level.data_to_latent(x, m)
            z_i, x = x.chunk(2, dim=1)
            if self.out_shape is not None:
                z.append(z_i.reshape((x.shape[0], -1) + self.out_shape))
            else:
                z.append(z_i.flatten(start_dim=1))
            log_det += level_log_det

        z_l, level_log_det = self.levels[-1].data_to_latent(x, m)
        if self.out_shape is not None:
            z.append(z_l.reshape((x.shape[0], -1) + self.out_shape))
        else:
            z.append(z_l.flatten(start_dim=1))
        log_det += level_log_det

        return torch.cat(z, dim=1), log_det

    def latent_to_data(self, z, m):
        if self.out_shape is not None:
            z = z.flatten(start_dim=1)

        log_det = 0.0
        b, c = z.shape

        # z is flattened: compute the number of entries to split off and concat to x
        split_size = c // 2**(self.l-1)

        # Compute the shape of x before it passes through the last level of the flow
        dim = int(np.sqrt(split_size // self.x_channels // 2**(self.l+1)))
        x = torch.Tensor(b, 0, dim, dim).to(z.device)

        # At each level, split out a new sequence of entries from z,
        # concatenate to x, and pass x through the level's operations
        z, x = self._split_z_and_concat_to_x(z, x, split_size)
        x, level_log_det = self.levels[-1].latent_to_data(x, m)
        log_det += level_log_det

        for level in reversed(self.levels[:-1]):
            z, x = self._split_z_and_concat_to_x(z, x, split_size)
            x, level_log_det = level.latent_to_data(x, m)
            log_det += level_log_det
            split_size *= 2

        return x, log_det

    @staticmethod
    def _split_z_and_concat_to_x(z, x, split_size):
        b, _, h, w = x.shape

        z, z_i = z.split((z.shape[1] - split_size, split_size), dim=1)
        x_i = z_i.view(b, -1, h, w)
        x = torch.cat((x_i, x), dim=1)

        return z, x
