import torch
import torch.nn as nn
import torch.nn.functional as f

from .base import FlowComponent


class ConditionalScale(FlowComponent):
    '''
    Flow component of the type z = mean(x) < 0? lx: rx

    l and r must be positive scalars.
    '''

    invertible = False
    conformal = True

    def __init__(self):
        super().__init__()
        self.left = nn.Parameter(torch.zeros(1)[0])
        self.right = nn.Parameter(torch.zeros(1)[0])

    def _left_const(self):
        return 2 * torch.sigmoid(self.left)

    def _right_const(self):
        return 2 * torch.sigmoid(self.right)

    def data_to_latent(self, x, m):
        cond = torch.mean(x, axis=(-3, -2, -1)) < 0

        z = torch.where(
            cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(x.shape),
            x * self._left_const(),
            x * self._right_const())
        log_det = torch.where(
            cond,
            m * torch.log(self._left_const()).expand(x.shape[0]),
            m * torch.log(self._right_const()).expand(x.shape[0]))

        return z, log_det


    def latent_to_data(self, z, m):
        cond = torch.mean(z, axis=(-3, -2, -1)) < 0

        x = torch.where(
            cond.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(z.shape),
            z / self._left_const(),
            z / self._right_const())
        log_det = torch.where(
            cond,
            m * torch.log(self._left_const()).expand(z.shape[0]),
            m * torch.log(self._right_const()).expand(z.shape[0]))

        return x, log_det


class Discontinuity(FlowComponent):
    '''
    Flow component of the type x = z < 0? z-1: z.

    This leaves a gap in the left-inverse which is filled by a constant.
    '''

    invertible = False
    conformal = True

    def __init__(self):
        super().__init__()
        # Keep constants so they're transferred to GPU when the module is
        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        self.gap_height = nn.Parameter(torch.Tensor([0.001])[0], requires_grad=False)

    def data_to_latent(self, x, m):
        z = torch.where(
            x < 0,
            torch.where(x < -1, x+1, self.gap_height),
            x)
        log_det = self.log_det.expand(z.shape[0])
        return z, log_det

    def latent_to_data(self, z, m):
        x = torch.where(z < 0, z-1, z)
        log_det = self.log_det.expand(z.shape[0])
        return x, log_det


class Inversion(FlowComponent):
    '''
    Flow component of the type x = z/||z||^2.

    This is numerically unstable.
    '''

    invertible = True
    conformal = True

    def data_to_latent(self, x, m):
        z = x / torch.sum(x**2, dim=(-1, -2, -3))
        log_det = -2 * m * torch.log(torch.norm(x))
        return z, log_det

    def latent_to_data(self, z, m):
        x = z / torch.sum(z**2, dim=(-1, -2, -3))
        log_det = 2 * m * torch.log(torch.norm(z))
        return x, log_det


class Scale(FlowComponent):

    invertible = True
    conformal = True

    def __init__(self):
        super().__init__()
        self.c = nn.Parameter(torch.zeros(1)[0])

    def _get_scale(self):
        # Multiplying by a raw constant has been unstable during training
        return 2 * torch.sigmoid(self.c)

    def data_to_latent(self, x, m):
        z =  x * self._get_scale()
        log_det = m * torch.log(torch.abs(self._get_scale())).expand(x.shape[0])
        return z, log_det

    def latent_to_data(self, z, m):
        x =  z / self._get_scale()
        log_det = m * torch.log(torch.abs(self._get_scale())).expand(z.shape[0])
        return x, log_det


class Shift(FlowComponent):

    invertible = True
    conformal = True

    def __init__(self, shape):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(shape))
        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)

    def initialize(self, x):
        self.b.data = -x.mean(axis=0)
        z, _ = self.data_to_latent(x, 0)
        return z

    def data_to_latent(self, x, m):
        z = x + self.b
        log_det = self.log_det.expand(z.shape[0])
        return z, log_det

    def latent_to_data(self, z, m):
        x = z - self.b
        log_det = self.log_det.expand(z.shape[0])
        return x, log_det


class SpecialConformal(FlowComponent):
    '''
    Flow component of the type x = (z- b ||z||^2)/(1 - 2bz + ||b||^2||z||^2).
    '''

    invertible = True
    conformal = True

    def __init__(self, shape):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(shape))

    def _get_parameter(self):
        # Norm of b can grow large in training causing instability
        # Scale vector elements into domain (-1, 1)
        return self.b

    def data_to_latent(self, x, m):
        z = ((x - self._get_parameter() * torch.sum(x**2, dim=(1, 2, 3), keepdim=True)) /
             (1 - 2 * torch.sum(self._get_parameter() * x, dim=(1, 2, 3), keepdim=True)
              + torch.sum(self._get_parameter()**2) * torch.sum(x**2, dim=(1, 2, 3), keepdim=True)))
        log_det = m * torch.log(
            torch.abs(1 - 2 * torch.sum(self._get_parameter() * x, dim=(1, 2, 3))
                      + torch.sum(self._get_parameter() ** 2) * torch.sum(x ** 2, dim=(-1, -2, -3))))
        return z, log_det

    def latent_to_data(self, z, m):
        x = ((z + self._get_parameter() * torch.sum(z**2, dim=(1, 2, 3), keepdim=True)) /
             (1 + 2 * torch.sum(self._get_parameter() * z, dim=(1, 2, 3), keepdim=True)
              + torch.sum(self._get_parameter()**2) * torch.sum(z**2, dim=(1, 2, 3), keepdim=True)))
        log_det = - m * torch.log(
            torch.abs(1 + 2 * torch.sum(self._get_parameter() * z, dim=(1, 2, 3))
                      + torch.sum(self._get_parameter() ** 2) * torch.sum(z ** 2, dim=(-1, -2, -3))))
        return x, log_det


class Pad(FlowComponent):

    invertible = False
    conformal = True

    def __init__(self, x_channels: int, z_channels: int):
        super().__init__()
        assert z_channels < x_channels
        self.z_channels = z_channels
        self.x_channels = x_channels
        self._channel_diff = x_channels - z_channels
        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)

    def data_to_latent(self, x, m):
        z =  torch.clone(x[...,:self.z_channels,:,:])
        log_det = self.log_det.expand(z.shape[0])
        return z, log_det

    def latent_to_data(self, z, m):
        x = f.pad(z, pad=(0, 0, 0, 0, 0, self._channel_diff))
        log_det = self.log_det.expand(z.shape[0])
        return x, log_det


class ActNorm(FlowComponent):

    invertible = True
    conformal = False

    def __init__(self, channels: int, scale_init=None, shift_init=None):
        super().__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def initialize(self, x):
        self.scale.data = 1/(x.std(axis=(0, 2, 3)).reshape(1, self.channels, 1, 1) + 1e-6)
        self.shift.data = -x.mean(axis=(0, 2, 3)).reshape(1, self.channels, 1, 1)
        z, _ = self.data_to_latent(x, 0)
        return z

    def data_to_latent(self, x, m):
        b, _, h, w = x.shape
        z = self.scale * (x + self.shift)
        log_det = h * w * torch.sum(torch.log(torch.abs(self.scale))).expand(b)
        return z, log_det

    def latent_to_data(self, z, m):
        b, _, h, w = z.shape
        x = z / self.scale - self.shift
        log_det = h * w * torch.sum(torch.log(torch.abs(self.scale))).expand(b)
        return x, log_det


class Squeeze(FlowComponent):

    invertible = True
    conformal = True

    def __init__(self, squeeze_factor=2):
        super().__init__()
        self.log_det = nn.Parameter(torch.zeros(1)[0], requires_grad=False)
        self.squeeze_factor = squeeze_factor

    def data_to_latent(self, x, m):
        b, c, h, w = x.shape
        factor = self.squeeze_factor

        x = x.view(b, c, h // factor, factor, w // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = x.view(b, c * factor**2, h // factor, w // factor)
        log_det = self.log_det.expand(z.shape[0])

        return z, log_det

    def latent_to_data(self, z, m):
        b, c, h, w = z.shape
        factor = self.squeeze_factor

        z = z.view(b, c // factor**2, factor, factor, h, w)
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = z.view(b, c // factor**2, h * factor, w * factor)
        log_det = self.log_det.expand(z.shape[0])

        return x, log_det
