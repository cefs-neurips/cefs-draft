import warnings
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn


class FlowComponent(nn.Module):
    '''
    Base class for a CEF component, which represents either a diffeomorphism
    or an almost-everywhere smooth embedding and its left inverse.

    This map is f:Z -> X, where dim(Z) <= dim(X).

    A FlowComponent should contain the two boolean properties below,
    which indicate whether map is invertible and almost-everywhere
    conformal, respectively.
    '''

    invertible: bool
    conformal: bool

    def forward(self, vec, inverse=False, **kwargs):
        if inverse:
            z, _ = self.data_to_latent(vec, m=0, **kwargs)
            return z
        else:
            x, _ = self.latent_to_data(vec, m=0, **kwargs)
            return x

    def initialize(self, x, **kwargs):
        '''Initial pass for weight initialization'''
        x, _ = self.data_to_latent(x, m=0, **kwargs)
        return x

    @abstractmethod
    def data_to_latent(self, x, m, **kwargs):
        '''
        Returns a tuple (z, log_det).
        - z is the output of this layer.
        - log_det is a tensor of shape [n_batches] indicating the log-det-Jacobian term
          incurred at x by this component.

        The log-det-Jacobian term for an arbitrary smooth embedding is
        -(1/2) log det J_f^T J_f.

        1) If this component is invertible, this method should return -log det J_f.
        2) Otherwise, this component should be a conformal embedding, so it has
           Jacobian J_f = cU, where U is orthogonal. This method should then return - m log|c|.
        '''
        pass

    @abstractmethod
    def latent_to_data(self, z, **kwargs):
        '''
        Returns a tuple (z, log_det).
        - x is the output of this layer.
        - log_det is a tensor of shape [n_batches] indicating the log-det-Jacobian term
          incurred at z by this component.
        '''
        pass

    def retract_orthogonal_params(self):
        '''After an optimizer step, retract orthogonal parameters onto the Stiefel manifold'''
        for parameter in self.parameters():
            if isinstance(parameter, OrthogonalParameter):
                parameter.retract()


class Sequential(FlowComponent):
    '''Sequentially construct a flow in the X -> Z direction.'''

    def __init__(self, *subcomponents):
        super().__init__()
        self.components = nn.ModuleList(subcomponents)
        self._check_structure()

    def initialize(self, x):
        for component in self.components:
            x = component.initialize(x)
        return x

    def data_to_latent(self, x, m):
        log_det = 0.0
        for component in self.components:
            x, component_log_det = component.data_to_latent(x, m)
            log_det += component_log_det
        return x, log_det

    def latent_to_data(self, z, m):
        log_det = 0.0
        for component in self.components[::-1]:
            z, component_log_det = component.latent_to_data(z, m)
            log_det += component_log_det
        return z, log_det

    def _check_structure(self):
        self.is_cef = True
        low_dimensional = True

        for comp in self.components[::-1]:
            # Make sure network is structured properly
            if not low_dimensional or not comp.invertible:
                low_dimensional = False
                if not comp.conformal:
                    self.is_cef = False
                    warnings.warn('Head of components list is not conformal. This is not a CEF. '
                                  'Densities will not be tractable.')

    def __add__(self, other: 'Sequential') -> 'Sequential':
        components = list(self.components) + list(other.components)
        return Sequential(*components)
