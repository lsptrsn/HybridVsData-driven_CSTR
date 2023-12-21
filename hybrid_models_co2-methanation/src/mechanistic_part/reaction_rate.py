#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:15:30 2022.

@author: peterson
"""
# extern
import joblib
import numpy as np
from scipy import constants
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

# dictionaries
from mechanistic_part.mechanistic_parameter import (LHHW_params,
                                                    reactor_params,
                                                    PL_params)
from mechanistic_part.mechanistics import get_partial_pressure, get_conversion


eps = np.finfo(float).eps


def LHHW(temp, p_CH4, p_CO2, p_H2O, p_H2):
    """Calculate the reaction rate via LHHW according to Koschany."""
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
    RG = constants.R*1e-3

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
    # Molar Reaction Rate in mol mcat^-3 s^-1
    reaction_rate_molar = reaction_rate_mass*density_cat
    return reaction_rate_molar


def power_law(params, temp, p_CH4, p_CO2, p_H2O, p_H2):
    """Calculate the reaction rate via power law according to Koschany."""
    density_cat = reactor_params['density_cat']
    temp_ref = PL_params['temp_ref']
    # Universal gas constant in kJ mol^-1 K^-1
    RG = constants.R*1e-3
    # get reaction constants
    # energy_act in J mol^-1, k0 in mol gcat^-1 bar^-XX s^-1
    k0, energy_act, n_CO2, n_H2 = params
    # Approximated by an empirical formula (Koschany2016 Eq. 30)
    Keq = 137*temp**(-3.998)*np.exp(158.7/RG/temp)
    # Equilibrium Constant in bar^-2
    Keq = Keq*1.01325**-2
    # Arrhenius Equation
    k = k0*np.exp(energy_act/RG*(1/temp_ref-1/temp))

    # Power Law
    r_faktor1 = k*p_H2**n_H2*p_CO2**n_CO2
    r_faktor2 = 1-(p_CH4*p_H2O**2/p_H2**4/p_CO2/Keq)
    # Mass based Reaction Rate in mol gcat^-1 s^-1
    reaction_rate = r_faktor1*r_faktor2

    # Converting the raction rate
    # Molar Reaction Rate in mol mcat^-3 s^-1
    reaction_rate_molar = reaction_rate*density_cat
    return reaction_rate_molar


def objective(params, temp, partial_pressure_CH4, partial_pressure_CO2,
              partial_pressure_H2O, partial_pressure_H2, time_residence,
              portion_product_in_feed, y):
    """Define objective function for getting PL values."""
    reaction_rate = power_law(params,
                              temp,
                              partial_pressure_CH4,
                              partial_pressure_CO2,
                              partial_pressure_H2O,
                              partial_pressure_H2)
    conv_CO2 = get_conversion(time_residence, portion_product_in_feed,
                              reaction_rate)
    mse = mean_squared_error(y, conv_CO2.flatten())
    return mse


def fit_params_PL(X):
    """Optimize objective function to fit PL parameter."""
    temp = X['temperature in K']
    try:
        partial_pressure = get_partial_pressure(X['pressure in bar'],
                                                X['portion of product in feed'],
                                                X['conversion CO2'])
        partial_pressure_CH4 = partial_pressure.iloc[:, 0],
        partial_pressure_CH4 = np.array(partial_pressure_CH4).flatten()
        partial_pressure_CO2 = partial_pressure.iloc[:, 1],
        partial_pressure_CO2 = np.array(partial_pressure_CO2).flatten()
        partial_pressure_H2O = partial_pressure.iloc[:, 2],
        partial_pressure_H2O = np.array(partial_pressure_H2O).flatten()
        partial_pressure_H2 = partial_pressure.iloc[:, 3]
        partial_pressure_H2 = np.array(partial_pressure_H2).flatten()
    except KeyError:
        partial_pressure_CH4 = X['partial pressure CH4 in bar']
        partial_pressure_CO2 = X['partial pressure CO2 in bar']
        partial_pressure_H2O = X['partial pressure H2O in bar']
        partial_pressure_H2 = X['partial pressure H2 in bar']
    time_residence = X['residence time in s']
    portion_product_in_feed = X['portion of product in feed']
    y = X['conversion CO2']
    #  k0, energy_act, m_CO2, m_H2
    bounds = [(1e-10, 10), (10, 1000), (0, 5), (0, 5)]
    print('Start parameter fitting')
    x0 = [(6.41e-05, 93.6, 0.31, 0.16)]
    res = minimize(objective, x0,
                   args=(np.array(temp),
                         np.array(partial_pressure_CH4),
                         np.array(partial_pressure_CO2),
                         np.array(partial_pressure_H2O),
                         np.array(partial_pressure_H2),
                         np.array(time_residence),
                         np.array(portion_product_in_feed),
                         np.array(y)),
                   method='Nelder-Mead'
                   )
    
    print('Ended parameter fitting')
    joblib.dump(res.x, '../models/PL_params.pkl')
    # print('Result: ', res)
    return res
