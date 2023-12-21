#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 08:20:36 2022

@author: peterson
"""
# extern
import casadi as c
import numpy as np

# functions

# dictionaries
from parameter_input import input_params, reactor_params
from reaction_rate import LHHW, power_law, reaction_rate_ME1
from utils import constants

eps = np.finfo(float).eps


def mass_energy_balance():
    """Calculate the mass and energy balance."""
    # Define symbolic variables
    mass_frac_out_CO2 = c.MX.sym("mass_frac_out_CO2")
    temp = c.MX.sym("temp")
    temp_in = c.MX.sym("temp_in")
    pressure_in = c.MX.sym("pressure_in")
    time_residence = c.MX.sym("time_residence")
    mole_frac_in_vec = c.MX.sym("mole_frac_in_vec", 5)
    para = c.vertcat(temp_in, pressure_in, time_residence, mole_frac_in_vec)
    mole_frac_in_vec = mole_frac_in_vec.T

    # Unpack Dictionary of Parameters
    stoic_vec = input_params['stoic_vec']
    density_gas_in = input_params['density_gas_in']
    heat_cap_in = input_params['heat_cap_in']

    # Unpack Dictionary of Reactor Specifications
    void_frac_bed = reactor_params['void_frac_bed']
    factor_eff = reactor_params['factor_eff']
    length_reactor = reactor_params['length_reactor']
    radius_reactor = reactor_params['radius_reactor']
    coef_heat_transfer = reactor_params['coef_heat_transfer']

    # Molar Mass in kg mol^-1
    MOLMASS_VEC = constants['MOLMASS_VEC']
    MOLMASS_CO2 = MOLMASS_VEC[0, 1]
    DH_METH = constants['DH_METH']

    # Inlet Mass fraction of all Species
    mass_frac_in_vec = mole_frac_in_vec*MOLMASS_VEC \
        / (c.cumsum(mole_frac_in_vec*MOLMASS_VEC)[-1]+eps)
    # mass_frac_in_vec = mole_frac_in_vec*MOLMASS_VEC \
    #    / (np.sum([mole_frac_in_vec*MOLMASS_VEC], axis=2).flatten()[:, None])

    # sanity check
    # assert np.isclose(np.sum(mass_frac_in_vec), 1, rtol=0.01)

    # Inlet concentrations of all species
    conc_in_vec = mole_frac_in_vec*density_gas_in \
        / (c.cumsum(mole_frac_in_vec*MOLMASS_VEC)[-1]+eps)
    # conc_in_vec = mole_frac_in_vec*density_gas_in \
    #     / (np.sum([mole_frac_in_vec*MOLMASS_VEC], axis=2).flatten()[:, None])

    # Shortcut for CO2
    mass_frac_in_CO2 = mass_frac_in_vec[0, 1]
    conc_in_CO2 = conc_in_vec[0, 1]
    stoic_CO2 = stoic_vec[0, 1]

    # Conversion of CO2
    conv_CO2 = (mass_frac_in_CO2-mass_frac_out_CO2)/mass_frac_in_CO2

    # Molar Concentration of all Species in mol m^-3
    conc_vec = conc_in_vec-conv_CO2*conc_in_CO2 \
        * stoic_vec/stoic_CO2

    # Mole Fraction of all Species
    mole_frac_vec = conc_vec/c.cumsum(conc_vec)[-1]
    # sanity check
    # assert np.isclose(np.sum(mole_frac_vec), 1, rtol=0.01)

    # Partial Pressure of all Species
    partial_pressure_vec = mole_frac_vec * pressure_in
    # sanity check
    # assert np.isclose(np.sum(partial_pressure_vec), pressure_in, rtol=0.01)

    # Pack u_vec
    r_vec = c.MX.zeros((1, 6))
    r_vec[:, 0:-1] = partial_pressure_vec
    r_vec[:, -1] = temp

    # Molar Reaction Rate in mol mcat^-3 s^-1
    reaction_rate = LHHW(r_vec)
    # Production Rate of every Species
    # Reaction source term in mol m^-3 s^-1
    reaction_source_term = -(1-void_frac_bed)*factor_eff*reaction_rate
    r_CO2 = stoic_CO2 * reaction_source_term

    # Mass Balance
    dwCO2dt = c.MX.sym('dwCO2dt')
    dwCO2dt = conv_CO2-((time_residence*r_CO2)/(void_frac_bed*conc_in_CO2+eps))

    # Stanton Number
    velocity_gas_in = void_frac_bed*length_reactor/(time_residence+eps)
    stanton_const = 2*length_reactor/(heat_cap_in*density_gas_in *
                                      radius_reactor+eps)
    stanton = coef_heat_transfer*stanton_const/(velocity_gas_in+eps)

    # Adiabatic Temperature Rise in K
    temp_rise_adiabatic = (mass_frac_out_CO2*(-DH_METH)
                           / (MOLMASS_CO2*heat_cap_in+eps))

    # Energy balance
    dmcpTdt = c.MX.sym('dmcpTdt')
    # Top = Tin = Tcool
    dmcpTdt = -conv_CO2+(((1+stanton)/temp_rise_adiabatic)*(temp-temp_in))
    return mass_frac_out_CO2, temp, para, dwCO2dt, dmcpTdt
