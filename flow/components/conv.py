import geotorch
import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import torch.nn.functional as f
from scipy.stats import ortho_group

from .base import FlowComponent


class OrthogonalDepthwiseConv(FlowComponent):

    invertible = True
    conformal = True

    def __init__(self, x_channels, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.x_channels = x_channels
        self.z_channels = x_channels * kernel_size**2
        self.groups = x_channels

        self.weights = []
        for _ in range(self.groups):
            self.weights.append(
                nn.Parameter(torch.Tensor(kernel_size**2, 1, kernel_size, kernel_size)))
        self.parameter_list = nn.ParameterList(self.weights)
        for i in range(self.groups):
            geotorch.orthogonal(self, f'parameter_list.{i}')

        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)

    def data_to_latent(self, x, m):
        assert x.shape[-2] % self.kernel_size == 0
        assert x.shape[-1] % self.kernel_size == 0

        log_det = self.log_det.expand(x.shape[0])
        conv_filter = torch.cat(self.weights)
        return f.conv2d(x, conv_filter, stride=self.kernel_size, groups=self.groups), log_det

    def latent_to_data(self, z, m):
        log_det = self.log_det.expand(z.shape[0])
        conv_filter = torch.cat(self.weights)
        return (f.conv_transpose2d(z, conv_filter, stride=self.kernel_size, groups=self.groups),
                log_det)


class Orthogonal1x1Conv(FlowComponent):

    invertible = True
    conformal = True

    def __init__(self, x_channels):
        super().__init__()
        self.x_channels = x_channels
        self.kernel = nn.Parameter(torch.Tensor(x_channels, x_channels))
        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        geotorch.orthogonal(self, 'kernel')

    def data_to_latent(self, x, m):
        log_det = self.log_det.expand(x.shape[0])
        return f.conv2d(x, self.kernel[...,None,None]), log_det

    def latent_to_data(self, z, m):
        log_det = self.log_det.expand(z.shape[0])
        return f.conv_transpose2d(z, self.kernel[...,None,None]), log_det


class OrthogonalGrouped1x1Conv(FlowComponent):

    invertible = True
    conformal = True

    def __init__(self, x_channels, kernel_depth=16):
        super().__init__()
        assert x_channels % kernel_depth == 0
        self.groups = x_channels // kernel_depth

        self.weights = [nn.Parameter(torch.Tensor(kernel_depth, kernel_depth, 1, 1))
                        for i in range(self.groups)]
        self.parameter_list = nn.ParameterList(self.weights)
        for i in range(self.groups):
            geotorch.orthogonal(self, f'parameter_list.{i}')

        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)

    def data_to_latent(self, x, m):
        log_det = self.log_det.expand(x.shape[0])
        conv_filter = torch.cat(self.weights)
        return f.conv2d(x, conv_filter, groups=self.groups), log_det

    def latent_to_data(self, z, m):
        log_det = self.log_det.expand(z.shape[0])
        conv_filter = torch.cat(self.weights)
        return f.conv_transpose2d(z, conv_filter, groups=self.groups), log_det


class HouseholderConv(FlowComponent):
    '''
    Convolution whose filter is parameterized by a householder matrix.
    '''

    invertible = True
    conformal = True

    def __init__(self, x_channels, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.x_channels = x_channels
        self.z_channels = x_channels * kernel_size**2
        self.matrix_size = self.z_channels

        self.v = nn.Parameter(torch.Tensor(self.matrix_size, 1)) # Householder vector
        torch.nn.init.normal_(self.v, std=0.01)

        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        self.identity = nn.Parameter(torch.eye(self.matrix_size), requires_grad=False)

    def _get_filter(self):
        conv_filter = self.identity - 2 * self.v @ self.v.T / torch.sum(self.v**2)
        return conv_filter.view(
            self.z_channels, self.x_channels, self.kernel_size, self.kernel_size)

    def data_to_latent(self, x, m):
        assert x.shape[-2] % self.kernel_size == 0
        assert x.shape[-1] % self.kernel_size == 0

        log_det = self.log_det.expand(x.shape[0])
        conv_filter = self._get_filter()
        return f.conv2d(x, conv_filter, stride=self.kernel_size), log_det

    def latent_to_data(self, z, m):
        log_det = self.log_det.expand(z.shape[0])
        conv_filter = self._get_filter()
        return f.conv_transpose2d(z, conv_filter, stride=self.kernel_size), log_det


class ConditionalConv(FlowComponent):
    '''
    Flow component of the type z = ||x|| >= 1? conv1(x): conv2(x)

    Both convs here are orthogonal and hence norm-preserving.
    '''

    invertible = True
    conformal = True

    def __init__(self, x_channels, kernel_depth=None):
        super().__init__()

        if kernel_depth is None:
            kernel_depth = x_channels

        self.conv_outer = Orthogonal1x1Conv(x_channels)
        self.conv_inner = Orthogonal1x1Conv(x_channels)

        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)

    def data_to_latent(self, x, m):
        cond = torch.sum(x**2, axis=(-3, -2, -1)) >= 1
        return (torch.where(cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(x.shape),
                            self.conv_outer(x, inverse=True),
                            self.conv_inner(x, inverse=True)),
                self.log_det.expand(x.shape[0]))


    def latent_to_data(self, z, m):
        cond = torch.sum(z**2, axis=(-3, -2, -1)) >= 1
        return (torch.where(cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(z.shape),
                            self.conv_outer(z),
                            self.conv_inner(z)),
                self.log_det.expand(z.shape[0]))


class ConditionalHouseholderConv(FlowComponent):
    '''
    Flow component of the type z = ||x|| >= 1? conv1(x): conv2(x)

    Both convs here are Householder and hence norm-preserving.
    '''

    invertible = True
    conformal = True

    def __init__(self, x_channels, kernel_size=2):
        super().__init__()

        self.conv_outer = HouseholderConv(x_channels, kernel_size)
        self.conv_inner = HouseholderConv(x_channels, kernel_size)

        self.kernel_size = kernel_size
        self.x_channels = x_channels
        self.z_channels = self.conv_outer.z_channels

        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)

    def data_to_latent(self, x, m):
        z_shape = (x.shape[0], self.z_channels,
                   x.shape[2] // self.kernel_size, x.shape[3] // self.kernel_size)

        cond = torch.sum(x**2, axis=(-3, -2, -1)) >= 1
        return (torch.where(cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(z_shape),
                           self.conv_outer(x, inverse=True),
                           self.conv_inner(x, inverse=True)),
                self.log_det.expand(x.shape[0]))

    def latent_to_data(self, z, m):
        x_shape = (z.shape[0], self.x_channels,
                   z.shape[2] * self.kernel_size, z.shape[3] * self.kernel_size)

        cond = torch.sum(z**2, axis=(-3, -2, -1)) >= 1
        return (torch.where(cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(x_shape),
                            self.conv_outer(z),
                            self.conv_inner(z)),
                self.log_det.expand(z.shape[0]))


class AffineCoupling(FlowComponent):

    invertible = True
    conformal = False

    def __init__(self, x_channels, num_blocks=3, hidden_size=512, kernel_size=3, padding=1,
                 additive=False):
        super().__init__()

        # Mask half the channels along the channel dimension
        self.unmasked_channels = x_channels // 2
        self.masked_channels = x_channels - self.unmasked_channels
        self.additive = additive
        self.additive_log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)

        if additive:
            out_size = self.masked_channels
        else:
            out_size = x_channels

        self._net = nn.Sequential(
            nn.Conv2d(self.unmasked_channels, hidden_size, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(hidden_size, out_size, kernel_size, padding=padding),
        )

        # Zero the last layer's weights so this layer is initially an identity function
        self._net[-1].weight.data.zero_()
        self._net[-1].bias.data.zero_()

    def data_to_latent(self, x, m):
        x_a, x_b = x.split(self.masked_channels, -3)

        if self.additive:
            t = self._net(x_b)
            s = 1
            log_det = self.additive_log_det.expand(x.shape[0])
        else:
            log_s, t = self._net(x_b).split(self.masked_channels, -3)
            s = torch.sigmoid(log_s + 2) # Glow code uses sigmoid instead of exp
            log_det = torch.sum(torch.log(s), axis=(-1, -2, -3))

        y_a = s * (x_a + t)
        y_b = x_b
        return torch.cat([y_b, y_a], -3), log_det # Swap the channel order

    def latent_to_data(self, z, m):
        # De-swap the channel order
        y_b, y_a = z.split([self.unmasked_channels, self.masked_channels], -3)

        if self.additive:
            t = self._net(y_b)
            s = 1
            log_det = self.additive_log_det.expand(z.shape[0])
        else:
            log_s, t = self._net(y_b).split([self.unmasked_channels, self.masked_channels], -3)
            s = torch.sigmoid(log_s + 2) # Glow code uses sigmoid instead of exp
            log_det = torch.sum(torch.log(s), axis=(-1, -2, -3))

        x_a = y_a / s - t
        x_b = y_b
        return torch.cat([x_a, x_b], -3), log_det


class Invertible1x1Conv(FlowComponent):

    invertible = True
    conformal = False

    def __init__(self, channels, affine=True):
        super().__init__()
        self.channels = channels

        # Initialize W = PLU in numpy, then transfer to torch as parameters
        w = ortho_group.rvs(channels) # Random orthogonal matrix as per Glow paper
        p, l, u = la.lu(w)
        p, l, u = torch.Tensor(p), torch.Tensor(l), torch.Tensor(u)

        # Parameterize s, the diagonal of u, in the log of its absolute value
        self.s_sign = nn.Parameter(u.diag().sign(), requires_grad=False)
        self.log_s = nn.Parameter(torch.log(u.diag().abs()))

        self.p = nn.Parameter(p, requires_grad=False)
        self.l = nn.Parameter(l)
        self.u = nn.Parameter(u.triu(1))

        # Identity matrix for computing the filter (keep it as a parameter so
        # it stays on the same device as the rest of the model)
        self.identity = nn.Parameter(torch.eye(self.channels), requires_grad=False)

    def _get_weight(self):
        p = self.p
        l = self.l.tril(-1) + self.identity
        u = self.u.triu(1)
        s = self.s_sign * torch.exp(self.log_s)
        return p @ l @ (u + s.diag())

    def data_to_latent(self, x, m):
        b, _, h, w = x.shape
        weight = self._get_weight()

        z = f.conv2d(x, weight.unsqueeze(-1).unsqueeze(-1))
        log_det = h * w * torch.sum(self.log_s).expand(b)
        return z, log_det

    def latent_to_data(self, z, m):
        b, _, h, w = z.shape
        weight = self._get_weight().inverse()

        x = f.conv2d(z, weight.unsqueeze(-1).unsqueeze(-1))
        log_det = h * w * torch.sum(self.log_s).expand(b)
        return x, log_det


class ConformalReLU(FlowComponent):
    '''
    Flow component of the type y = ReLU([w -w]^T*x); as in Trumpets, but with Householder filters.
    "Trumpets: Injective Flows for Inference and Inverse Problems"
    by Konik Kothari et al.
    https://arxiv.org/pdf/2102.10461.pdf
    '''

    invertible = False
    conformal = True

    def __init__(self, x_channels, kernel_size=1):
        super().__init__()

        self.conv = HouseholderConv(x_channels // 2, kernel_size)

        self.kernel_size = kernel_size
        self.z_channels = self.conv.z_channels

    def data_to_latent(self, x, m):
        x_pos, x_neg = torch.chunk(x, 2, dim=1)
        z = self.conv(x_pos - x_neg, inverse=True)

        log_det = torch.zeros(x.shape[0]).to(z.device)
        return z, log_det

    def latent_to_data(self, z, m):
        top = self.conv(z)
        bottom = -top
        x = f.relu(torch.cat((top, bottom), axis=1))

        log_det = torch.zeros(z.shape[0]).to(z.device)
        return x, log_det
