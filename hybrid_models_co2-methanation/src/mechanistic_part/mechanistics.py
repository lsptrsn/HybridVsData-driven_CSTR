"""
Created on Mon Apr 04 15:07:36 2022.

@author: peterson
"""
from molmass import Formula
import numpy as np
import pandas as pd
from scipy import constants
import sys
import torch

from mechanistic_part.mechanistic_parameter import (input_params,
                                                    reactor_params)

eps = sys.float_info.epsilon


def get_reaction_rate(portion_product_in_feed, time_residence, conversion_CO2,
                      length_reactor=None):
    """Get the reaction rate at steady state with pseudo-experimental data."""
    # Unpack Dictionary of Input Parameters
    stoic_vec = input_params['stoic_vec']
    density_gas_in = input_params['density_gas_in']

    # Unpack Dictionary of Reactor Parameters
    void_frac_bed = reactor_params['void_frac_bed']
    if length_reactor is None:
        length_reactor = reactor_params['length_reactor']
    factor_eff = reactor_params['factor_eff']

    # Unpack Vector of stoichiometric coefficients
    stoic_CH4 = stoic_vec[0, 0]
    stoic_CO2 = stoic_vec[0, 1]
    stoic_H2O = stoic_vec[0, 2]
    stoic_H2 = stoic_vec[0, 3]

    # Molar Mass in kg mol^-1
    MOLMASS_VEC = np.zeros((1, 5))
    MOLMASS_VEC[0, 0] = Formula('CH4').mass*1e-3  # CH4
    MOLMASS_VEC[0, 1] = Formula('CO2').mass*1e-3  # CO2
    MOLMASS_VEC[0, 2] = Formula('H2O').mass*1e-3  # H2O
    MOLMASS_VEC[0, 3] = Formula('H2').mass*1e-3  # H2
    MOLMASS_VEC[0, 4] = Formula('N2').mass*1e-3  # N2

    # Calculate inlet mole fractions
    mole_frac_in = np.zeros((len(portion_product_in_feed), 5))
    # molar ratio H2 to CO2
    ratio_H2_CO2 = [stoic_H2, stoic_CO2]
    # ratio H2O to CH4
    ratio_H2O_CH4 = [stoic_H2O, stoic_CH4]
    # CH4
    mole_frac_in[:, 0] = portion_product_in_feed*(ratio_H2O_CH4[1]
                                                  / sum(ratio_H2O_CH4))
    # CO2
    mole_frac_in[:, 1] = (1-portion_product_in_feed)*(ratio_H2_CO2[1]
                                                      / sum(ratio_H2_CO2))
    # H2O
    mole_frac_in[:, 2] = portion_product_in_feed*(ratio_H2O_CH4[0]
                                                  / sum(ratio_H2O_CH4))
    # H2
    mole_frac_in[:, 3] = (1-portion_product_in_feed)*(ratio_H2_CO2[0]
                                                      / sum(ratio_H2_CO2))
    mole_frac_in[:, 4] = 0  # N2

    # Get inlet concentration in mol m⁻3
    concentration_in = mole_frac_in*density_gas_in \
        / (np.sum([mole_frac_in * MOLMASS_VEC], axis=2).flatten()[:, None])

    # Shortcuts for CO2
    concentration_in_CO2 = concentration_in[:, 1]

    # reactive source term in mol m^-3 s^-1
    reactive_source_term = conversion_CO2*concentration_in_CO2*void_frac_bed \
        / time_residence
    # molar reaction rate in mol mcat^-3 s^-1
    molar_reaction_rate = -reactive_source_term/(1-void_frac_bed) \
        / factor_eff/stoic_CO2
    return molar_reaction_rate


def get_partial_pressure(pressure, portion_product_in_feed, conversion_CO2):
    """Get parameter to estimate reaction rate."""
    # Unpack Dictionary of Parameters
    stoic_vec = input_params['stoic_vec']
    density_gas_in = input_params['density_gas_in']

    # Molar Mass in kg mol^-1
    MOLMASS_VEC = np.zeros((1, 5))
    MOLMASS_VEC[0, 0] = Formula('CH4').mass*1e-3  # CH4
    MOLMASS_VEC[0, 1] = Formula('CO2').mass*1e-3  # CO2
    MOLMASS_VEC[0, 2] = Formula('H2O').mass*1e-3  # H2O
    MOLMASS_VEC[0, 3] = Formula('H2').mass*1e-3  # H2
    MOLMASS_VEC[0, 4] = Formula('N2').mass*1e-3  # N2

    # Unpack Vector of stoichiometric coefficients
    stoic_CH4 = stoic_vec[0, 0]
    stoic_CO2 = stoic_vec[0, 1]
    stoic_H2O = stoic_vec[0, 2]
    stoic_H2 = stoic_vec[0, 3]
    # molar ratio H2 to CO2
    ratio_H2_CO2 = [stoic_H2, stoic_CO2]
    # ratio H2O to CH4
    ratio_H2O_CH4 = [stoic_H2O, stoic_CH4]

    # Calculate inlet mole fractions
    mole_frac_in = np.zeros((len(portion_product_in_feed), 5))
    mole_frac_in[:, 0] = portion_product_in_feed \
        * (ratio_H2O_CH4[1]/sum(ratio_H2O_CH4))
    # CO2
    mole_frac_in[:, 1] = (1-portion_product_in_feed) \
        * (ratio_H2_CO2[1]/sum(ratio_H2_CO2))
    # H2O
    mole_frac_in[:, 2] = portion_product_in_feed \
        * (ratio_H2O_CH4[0]/sum(ratio_H2O_CH4))
    # H2
    mole_frac_in[:, 3] = (1-portion_product_in_feed) \
        * (ratio_H2_CO2[0]/sum(ratio_H2_CO2))
    mole_frac_in[:, 4] = 0  # N2

    # Inlet concentration in mol m⁻3
    concentration_in = mole_frac_in*density_gas_in \
        / (np.sum([mole_frac_in * MOLMASS_VEC], axis=2).flatten()[:, None])

    # Shortcuts for CO2
    concentration_in_CO2 = concentration_in[:, 1]
    stoic_CO2 = stoic_vec[0, 1]

    # Molar Concentration of all Species in mol m^-3
    conv_conc_CO2 = conversion_CO2*concentration_in_CO2
    conv_conc_CO2 = np.array(conv_conc_CO2).reshape(-1, 1)
    concentration = concentration_in-conv_conc_CO2 \
        * stoic_vec/stoic_CO2

    # Mole Fraction of all Species
    conc_sum_per_exp = np.array(np.sum(concentration, axis=1)).reshape(-1, 1)
    mole_frac = concentration/conc_sum_per_exp

    # Partial Pressure of all Species
    partial_pressure = mole_frac * np.array(pressure).reshape(-1, 1)
    return pd.DataFrame(partial_pressure)


def get_conversion(time_residence, portion_product_in_feed, reaction_rate,
                   length_reactor=None):
    """Get conversion with estimated reaction rate."""
    # Unpack Dictionary of Reactor Specifications
    void_frac_bed = reactor_params['void_frac_bed']
    if length_reactor is None:
        length_reactor = reactor_params['length_reactor']
    factor_eff = reactor_params['factor_eff']
    stoic_vec = input_params['stoic_vec']
    density_gas_in = input_params['density_gas_in']

    # get inlet concentration of CO2
    # Molar Mass in kg mol^-1
    MOLMASS_VEC = np.zeros((1, 5))
    MOLMASS_VEC[0, 0] = Formula('CH4').mass*1e-3  # CH4
    MOLMASS_VEC[0, 1] = Formula('CO2').mass*1e-3  # CO2
    MOLMASS_VEC[0, 2] = Formula('H2O').mass*1e-3  # H2O
    MOLMASS_VEC[0, 3] = Formula('H2').mass*1e-3  # H2
    MOLMASS_VEC[0, 4] = Formula('N2').mass*1e-3  # N2

    # Unpack Vector of stoichiometric coefficients
    stoic_CH4 = stoic_vec[0, 0]
    stoic_CO2 = stoic_vec[0, 1]
    stoic_H2O = stoic_vec[0, 2]
    stoic_H2 = stoic_vec[0, 3]
    # molar ratio H2 to CO2
    ratio_H2_CO2 = [stoic_H2, stoic_CO2]
    # ratio H2O to CH4
    ratio_H2O_CH4 = [stoic_H2O, stoic_CH4]

    # Reaction source term in mol m^-3 s^-1
    reaction_source_term = -(1-void_frac_bed)*factor_eff*reaction_rate
    r_CO2 = stoic_CO2 * reaction_source_term

    # Calculate inlet mole fractions
    try:
        mole_frac_in = np.zeros((len(portion_product_in_feed), 5))
    except TypeError:
        mole_frac_in = np.zeros((1, 5))
    mole_frac_in = np.zeros((len(portion_product_in_feed), 5))
    mole_frac_in[:, 0] = portion_product_in_feed \
        * (ratio_H2O_CH4[1]/sum(ratio_H2O_CH4))
    # CO2
    mole_frac_in[:, 1] = (1-portion_product_in_feed) \
        * (ratio_H2_CO2[1]/sum(ratio_H2_CO2))
    # H2O
    mole_frac_in[:, 2] = portion_product_in_feed \
        * (ratio_H2O_CH4[0]/sum(ratio_H2O_CH4))
    # H2
    mole_frac_in[:, 3] = (1-portion_product_in_feed) \
        * (ratio_H2_CO2[0]/sum(ratio_H2_CO2))
    mole_frac_in[:, 4] = 0  # N2

    # Inlet concentration in mol m⁻3
    concentration_in = mole_frac_in*density_gas_in \
        / (np.sum([mole_frac_in * MOLMASS_VEC], axis=2).flatten()[:, None])
    # Shortcuts for CO2
    concentration_in_CO2 = concentration_in[:, 1]
    if r_CO2.dtype == torch.float64 or r_CO2.dtype == torch.float32:
        time_residence = time_residence.to(device="cuda")
        r_CO2 = r_CO2.to(device="cuda")
        numerator = time_residence*r_CO2
        denumerator = void_frac_bed*concentration_in_CO2+eps
        denumerator_torch = torch.from_numpy(denumerator)
        conv_CO2 = numerator/denumerator_torch.to(device="cuda")
    else:
        conv_CO2 = (time_residence*r_CO2) \
            / (void_frac_bed*concentration_in_CO2+eps)
    return conv_CO2


def concentrations_from_conversion(conc_in_vec, conv_CO2, stoic_vec):
    """Estimate concentration from conversion of CH4."""
    conc_in_vec = conc_in_vec.flatten()
    # Molar Concentration of all Species in mol m^-3
    conc_CH4 = conc_in_vec[0]-conv_CO2*conc_in_vec[1] \
        * stoic_vec[0, 0]/stoic_vec[0, 1]
    conc_CO2 = conc_in_vec[1]-conv_CO2*conc_in_vec[1] \
        * stoic_vec[0, 1]/stoic_vec[0, 1]
    conc_H2O = conc_in_vec[2]-conv_CO2*conc_in_vec[1] \
        * stoic_vec[0, 2]/stoic_vec[0, 1]
    conc_H2 = conc_in_vec[3]-conv_CO2*conc_in_vec[1] \
        * stoic_vec[0, 3]/stoic_vec[0, 1]
    conc_N2 = conc_in_vec[4]-conv_CO2*conc_in_vec[1] \
        * stoic_vec[0, 4]/stoic_vec[0, 1]
    conc_vec = np.array([conc_CH4, conc_CO2, conc_H2O, conc_H2, conc_N2])
    return conc_vec


def get_potential_term(X):
    """Calculate potential term of reaction rate."""
    partial_pressure_CH4 = X[:, 0]
    partial_pressure_CO2 = X[:, 1]
    partial_pressure_H2O = X[:, 2]
    partial_pressure_H2 = X[:, 3]
    temperature = X[:, 5]
    temperature = temperature
    # Universal gas constant in kJ mol^-1 K^-1
    RG = constants.R*1e-3
    # Approximated by an empirical formula (Koschany2016 Eq. 30)
    equilibrium_const = 137*temperature**(-3.998) \
        * np.exp(158.7/RG/temperature)
    # Equilibrium Constant in bar^-2
    equilibrium_const = equilibrium_const*1.01325**-2
    quotient = partial_pressure_CH4*partial_pressure_H2O**2 \
        / (partial_pressure_CO2*partial_pressure_H2**4*equilibrium_const+eps)
    potential_term = 1-quotient
    return potential_term
