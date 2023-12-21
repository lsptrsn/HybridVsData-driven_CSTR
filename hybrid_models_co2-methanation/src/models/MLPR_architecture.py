"""Neural Net Implementation with PyTorch."""
import numpy as np
import random
from scipy import constants
from skorch import NeuralNetRegressor
import sys
import torch
from torch import nn

from mechanistic_part.mechanistics import get_conversion, get_potential_term
from data.utils_data import unscale_data

eps = sys.float_info.epsilon

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

torch.set_default_dtype(torch.float32)

import math
def lecun_normal_(tensor: torch.Tensor) -> torch.Tensor:
    input_size = tensor.shape[-1] # Assuming that the weights' input dimension is the last.
    std = math.sqrt(1/input_size)
    with torch.no_grad():
        return tensor.normal_(-std,std)

class MLPR(nn.Module):
    """Class for creating Neural Net via pyTorch."""

    def __init__(self,
                 feature_count,
                 n_hidden_layer=1,
                 n_neurons_per_layer=29,
                 dropout_ratio=0.00,
                 ):
        """Initialize the layers I want to use."""
        super(MLPR, self).__init__()
        # number of hidden layer
        self.n_hidden_layer = n_hidden_layer
        # number of neurons per layer
        self.n_neurons_per_layer = n_neurons_per_layer
        # activation
        self.activation = nn.SELU()
        # dropout ratio
        self.dropout = nn.Dropout(dropout_ratio, inplace=False)
        # stack layers to net
        all_layers = []
        for i in range(self.n_hidden_layer):
            # input layer
            if i == 0:
                layer = nn.Linear(feature_count, self.n_neurons_per_layer)
                lecun_normal_(layer.weight)
            # hidden layers
            else:
                layer = nn.Linear(self.n_neurons_per_layer,
                                  self.n_neurons_per_layer)
                lecun_normal_(layer.weight)
            all_layers.append(layer)
            # all_layers.append(nn.BatchNorm1d(n_neurons_per_layer))
            all_layers.append(self.activation)
            all_layers.append(self.dropout)
        # output layer
        output_layer = nn.Linear(n_neurons_per_layer, 1)
        # lecun_normal_(layer.weight)
        all_layers.append(output_layer)
        self.model_stack = nn.Sequential(*all_layers)

    def forward(self, X, **kwargs):
        """Forward propagate through Neural Net."""
        X = self.model_stack(X.float())
        return X


class MyNeuralNetRegressor(NeuralNetRegressor):
    """Make y 2-dimensional."""

    def fit(self, X, y, **fit_params):
        """Use fit method with 2-dimensional y."""
        if isinstance(y, np.ndarray) and y.ndim == 1:
            y = y.reshape(-1, 1)
        out = super(MyNeuralNetRegressor, self).fit(X.astype(np.float64),
                                                    y.astype(np.float64),
                                                    **fit_params)
        return out


class MyMSELossFunction(nn.Module):
    """Run custom loss function for MSE."""

    def __init__(self, dw=None, ml_type=None):
        super(MyMSELossFunction, self).__init__()
        self.dw = dw
        self.ml_type = ml_type

    def forward(self, y_pred, target, X):
        """Calculate MSE."""
        # loss function
        loss_unreduced = ((target-y_pred)**2)
        loss_unreduced = loss_unreduced.to(device="cuda")
        if self.dw is not None:
            # get weights
            X = np.array(X.to(device="cpu"))
            if self.ml_type == 'datadriven':
                sample_weights = torch.from_numpy(self.dw([X[:, -1]])).to(
                    device="cuda")
                loss_reduced = sample_weights.reshape(-1, 1)*loss_unreduced
            elif self.ml_type == 'parallel':
                sample_weights = torch.from_numpy(self.dw([X[:, -1]])).to(
                    device="cuda")
                loss_reduced = sample_weights.reshape(-1, 1)*loss_unreduced
        else:
            loss_reduced = loss_unreduced
        return torch.mean(loss_reduced)


class MyHybridLossFunction(nn.Module):
    """Run custom loss function for hybrid modeling."""

    def __init__(self, dw=None):
        super(MyHybridLossFunction, self).__init__()
        self.dw = dw

    def forward(self, y_pred, target, X):
        """Calculate MSE."""
        # unscale partial pressure
        (partial_pressure_CH4, partial_pressure_CO2, partial_pressure_H2O,
         partial_pressure_H2, temperature_in, temperature, time_residence,
         portion_product_in_feed) = unscale_data(X)
        y_pred = y_pred.to(device="cuda")
        reaction_rate = y_pred
        conv_CO2 = get_conversion(time_residence, portion_product_in_feed,
                                  reaction_rate.flatten(), length_reactor=None)
        conv_CO2 = conv_CO2.reshape(-1, 1)
        # loss function
        conv_CO2 = conv_CO2.to(device="cuda")
        target = target.to(device="cuda")
        loss_unreduced = (target-conv_CO2) ** 2
        if self.dw is not None:
            target = np.array(target.to(device="cpu"))
            sample_weights = torch.from_numpy(self.dw([target])).to(
                device="cuda")
            loss_reduced = sample_weights.reshape(-1, 1)*loss_unreduced
        else:
            loss_reduced = loss_unreduced
        return torch.mean(loss_reduced)


class MyHybridEquillibriumLossFunction(nn.Module):
    """Run custom loss function for hybrid modeling."""

    def __init__(self, dw=None):
        super(MyHybridEquillibriumLossFunction, self).__init__()
        self.dw = dw

    def forward(self, y_pred, target, X):
        """Calculate MSE."""
        # unscale partial pressure
        (partial_pressure_CH4, partial_pressure_CO2, partial_pressure_H2O,
         partial_pressure_H2, temperature_in, temperature, time_residence,
         portion_product_in_feed) = unscale_data(X)
        # Universal gas constant in kJ mol^-1 K^-1
        RG = constants.R*1e-3
        # Approximated by an empirical formula (Koschany2016 Eq. 30)
        equilibrium_const = 137*temperature**(-3.998) \
            * np.exp(158.7/RG/temperature)
        # Equilibrium Constant in bar^-2
        equilibrium_const = equilibrium_const*1.01325**-2
        quotient = partial_pressure_CH4*partial_pressure_H2O**2 \
            / (partial_pressure_CO2*partial_pressure_H2**4
               * equilibrium_const+eps)
        potential_term = 1-quotient
        y_pred = y_pred.to(device="cuda")
        potential_term = potential_term.to(device="cuda")
        reaction_rate = y_pred*potential_term.reshape(-1, 1)
        conv_CO2 = get_conversion(time_residence, portion_product_in_feed,
                                  reaction_rate.flatten(), length_reactor=None)
        conv_CO2 = conv_CO2.reshape(-1, 1)
        # loss function
        conv_CO2 = conv_CO2.to(device="cuda")
        target = target.to(device="cuda")
        loss_unreduced = torch.mean((target-conv_CO2) ** 2)
        if self.dw is not None:
            target = np.array(target.to(device="cpu"))
            sample_weights = torch.from_numpy(self.dw([target])).to(device="cuda")
            loss_reduced = torch.mean(sample_weights.reshape(-1, 1)
                                      * loss_unreduced)
        else:
            loss_reduced = loss_unreduced
        return loss_reduced


def my_loss(y_true, y_pred, ml_type, X=None):
    """Get loss function for hybrid serial modeling."""
    if ml_type == 'serial_equilibrium':
        potential_term = get_potential_term(X)
        reaction_rate = y_pred.reshape(-1, 1)*potential_term.reshape(-1, 1)
    elif ml_type == 'serial':
        reaction_rate = y_pred.reshape(-1, 1)
    conv_CO2 = get_conversion(X[:, 6], X[:, 7], reaction_rate.flatten(),
                              length_reactor=None)
    conv_CO2 = conv_CO2.flatten()
    # loss function
    output_errors = np.mean((y_true - conv_CO2) ** 2)
    return output_errors
