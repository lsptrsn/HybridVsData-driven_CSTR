"""
Created on Mon Feb 14 08:16:08 2022.

@author: peterson
"""
# extern
import joblib
import numpy as np

# dictionaries
from parameter_input import PL_params, LHHW_params, reactor_params
from utils import constants

eps = np.finfo(float).eps


def power_law(r_vec):
    """Calculate the reaction rate via power law according to Koschany."""
    # Unpack u_vec
    p_CH4 = r_vec[0, 0]
    p_CO2 = r_vec[0, 1]
    p_H2O = r_vec[0, 2]
    p_H2 = r_vec[0, 3]
    temp = r_vec[0, -1]

    # sanity check
    # assert p_CO2 >= 0.0
    # assert p_H2 >= 0.0
    # assert p_CH4 >= 0.0
    # assert p_H2O >= 0.0
    # assert temp > 0.0

    # Unpack dictionary with parameters for PL
    temp_ref = PL_params['temp_ref']
    k0 = PL_params['k0']
    energy_act = PL_params['energy_act']
    n_H2 = PL_params['n_H2']
    n_CO2 = PL_params['n_CO2']
    eq_const_OH_0 = PL_params['eq_const_OH_0']
    dH_OH = PL_params['dH_OH']
    density_cat = reactor_params['density_cat']

    # Universal gas constant in kJ mol^-1 K^-1
    RG = constants['RG']
    # Eqilibrium Constant in Pa
    # Approximated by an empirical formula (Koschany2016 Eq. 30)
    Keq = 137*temp**(-3.998)*np.exp(158.7/RG/temp)
    # Equilibrium Constant in bar^-2
    Keq = Keq*1.01325**-2
    # Arrhenius Equation
    k = k0*np.exp(energy_act/RG*(1/temp_ref-1/temp))
    # van't Hoff Equation
    K_OH = eq_const_OH_0*np.exp(dH_OH/RG*(1/temp_ref-1/temp))

    # Power Law
    r_faktor1 = k*p_H2**n_H2*p_CO2**n_CO2/(1+K_OH*p_H2O/p_H2**0.5)
    r_faktor2 = 1-(p_CH4*p_H2O**2/p_H2**4/p_CO2/Keq)
    # Mass based Reaction Rate in mol gcat^-1 s^-1
    reaction_rate = r_faktor1*r_faktor2

    # Converting the raction rate
    # Molar Reaction Rate in mol mcat^-3 s^-1
    reaction_rate = reaction_rate*density_cat
    return reaction_rate


def LHHW(r_vec):
    """Calculate the reaction rate via LHHW according to Koschany."""
    # Unpack u_vec
    p_CH4 = r_vec[0, 0]
    p_CO2 = r_vec[0, 1]
    p_H2O = r_vec[0, 2]
    p_H2 = r_vec[0, 3]
    temp = r_vec[0, -1]

    # sanity check
    # assert p_CO2 >= 0.0
    # assert p_H2 >= 0.0
    # assert p_CH4 >= 0.0
    # assert p_H2O >= 0.0
    # assert temp > 0.0

    # Unpack dictionary with parameters for LHHW
    temp_ref = LHHW_params['temp_ref']
    k0 = LHHW_params['k0']
    energy_act = LHHW_params['energy_act']
    eq_const_H2_0 = LHHW_params['eq_const_H2_0']
    eq_const_OH_0 = LHHW_params['eq_const_OH_0']
    eq_const_mix_0 = LHHW_params['eq_const_mix_0']
    dH_H2 = LHHW_params['dH_H2']
    dH_OH = LHHW_params['dH_OH']
    dH_mix = LHHW_params['dH_mix']
    density_cat = reactor_params['density_cat']

    # Universal gas constant in kJ mol^-1 K^-1
    RG = constants['RG']

    # Eqilibrium Constant in Pa^-2
    # Approximated by an empirical formula (Koschany2016 Eq. 30)
    Keq = 137*temp**(-3.998)*np.exp(158.7/RG/temp)
    # Equilibrium Constant in bar^-2
    Keq = Keq*1.01325**-2

    # Arrhenius Equation
    k = k0*np.exp(energy_act/RG*(1/temp_ref-1/temp))
    k = 0.2*k
    # van't Hoff Equations
    K_H2 = eq_const_H2_0*np.exp(dH_H2/RG*(1/temp_ref-1/temp))
    K_OH = eq_const_OH_0*np.exp(dH_OH/RG*(1/temp_ref-1/temp))
    K_mix = eq_const_mix_0*np.exp(dH_mix/RG*(1/temp_ref-1/temp))

    # Reaction Rate
    r_NUM = k*p_H2**0.5*p_CO2**0.5*(1-(p_CH4*p_H2O**2)/(p_CO2*p_H2**4*Keq+eps))
    r_DEN = (1+K_OH*p_H2O/(p_H2**0.5+eps)+K_H2*p_H2**0.5+K_mix*p_CO2**0.5)
    # Mass based Reaction Rate in mol gcat^-1 s^-1
    reaction_rate_mass = r_NUM/(r_DEN**2+eps)

    # Converting the raction rate
    # Molar Reaction Rate in mol mcat^-3 s^-1
    reaction_rate_molar = reaction_rate_mass*density_cat

    return reaction_rate_molar

def reaction_rate_ME1(r_MLP_vec):
    model = joblib.load('MLP_ME1.pkl')
    model.regressor['algorithm'].loss = "squared_error"
    reaction_rate = model.predict(r_MLP_vec)
    reaction_rate = reaction_rate.reshape(-1, 1)
    return reaction_rate
