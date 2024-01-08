#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:28:04 2022

@author: peterson
"""
import casadi as c
import joblib
import numpy as np
import paths

from mechanistic_part.mechanistics import concentrations_from_conversion
from mechanistic_part.mechanistic_parameter import (input_params,
                                                    reactor_params,
                                                    constants)
from mechanistic_part.reaction_rate import power_law


def mass_energy_balance(model):
    """Calculate the mass and energy balance."""
    eps = np.finfo(float).eps
    # Define symbolic variables
    mass_frac_out_CO2 = c.MX.sym("mass_frac_out_CO2")
    temp = c.MX.sym("temp")
    temp_in = c.MX.sym("temp_in")
    pressure_in = c.MX.sym("pressure_in")
    time_residence = c.MX.sym("time_residence")
    mole_frac_in_vec = c.MX.sym("mole_frac_in_vec", 5)
    portion_product_in_feed = c.MX.sym("portion_product_in_feed")
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

    # Molar Mass of the Species in kg mol^-1
    MOLMASS_VEC = constants['MOLMASS_VEC']
    MOLMASS_CO2 = MOLMASS_VEC[0, 1]
    # Enthalpy Difference Methane in J mol^-1
    DH_METH = constants['DH_METH']

    # Inlet Mass fraction of all Species
    mass_frac_in_vec = mole_frac_in_vec*MOLMASS_VEC \
        / (c.cumsum(mole_frac_in_vec*MOLMASS_VEC)[-1]+eps)
    # mass_frac_in_vec = mole_frac_in_vec*MOLMASS_VEC \
    #    / (np.sum([mole_frac_in_vec*MOLMASS_VEC], axis=2).flatten()[:, None])

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

    # Partial Pressure of all Species
    partial_pressure_vec = mole_frac_vec * pressure_in

    # Input params
    p_CH4 = partial_pressure_vec[:, 0]
    p_CO2 = partial_pressure_vec[:, 1]
    p_H2O = partial_pressure_vec[:, 2]
    p_H2 = partial_pressure_vec[:, 3]
    p_N2 = 1e-10
    reaction_rate = 'PL'
    if reaction_rate == 'PL':
        PL_params = joblib.load(paths.model_path / 'PL_params.pkl')
        reaction_rate_PL = power_law(PL_params, temp,
                                     p_CH4, p_CO2, p_H2O, p_H2)
        X = [pressure_in, temp_in, time_residence, portion_product_in_feed]
        residual = model.predict(X)
        reaction_rate = reaction_rate_PL+residual
    elif reaction_rate == 'MLPR':
        X = [p_CH4, p_CO2, p_H2O, p_H2, p_N2, temp_in,
             time_residence, portion_product_in_feed]
        reaction_rate = model.predict(X)
    # Production Rate of every Species
    # Reaction source term in mol m^-3 s^-1
    reaction_source_term = -(1-void_frac_bed)*factor_eff*reaction_rate
    r_CO2 = stoic_CO2 * reaction_source_term

    # Mass Balance
    dwCO2dt = c.MX.sym('dwCO2dt')
    dwCO2dt = conv_CO2-((time_residence*r_CO2)/(void_frac_bed*conc_in_CO2+eps))

    # Stanton Number
    time_residence = void_frac_bed*length_reactor/(time_residence+eps)
    stanton_const = 2*length_reactor/(heat_cap_in*density_gas_in *
                                      radius_reactor+eps)
    stanton = coef_heat_transfer*stanton_const/(time_residence+eps)

    # Adiabatic Temperature Rise in K
    temp_rise_adiabatic = (mass_frac_out_CO2*(-DH_METH)
                           / (MOLMASS_CO2*heat_cap_in+eps))

    # Energy balance
    dmcpTdt = c.MX.sym('dmcpTdt')
    # Top = Tin = Tcool
    dmcpTdt = -conv_CO2+(((1+stanton)/temp_rise_adiabatic)*(temp-temp_in))
    return mass_frac_out_CO2, temp, para, dwCO2dt, dmcpTdt


def solve_DAE(model, X):
    """Solve DAE to get steady state data."""
    # Unpack Argument
    pressure_vec = X[:, 0]
    temp_in_vec = X[:, 1]
    time_residence_vec = X[:, 2]
    portion_product_in_feed_vec = X[:, 3]
    # Unpack Dictionary of Parameters
    stoic_vec = input_params['stoic_vec']
    density_gas_in = input_params['density_gas_in']
    # Molar Mass in kg mol^-1
    MOLMASS_VEC = constants['MOLMASS_VEC']
    # empty dicts to save resulst
    conv_CO2_all = []
    temp_in_all = []
    temp_all = []
    experiment_no = []
    # Solve DAE
    [mass_frac_out_CO2, temp, para_sym,
     dwCO2dt, dmcpTdt] = mass_energy_balance(model)
    # options
    opts = {}
    opts["expand"] = True
    opts["ipopt.max_iter"] = 100000
    opts["ipopt.print_level"] = 0
    # opts["ipopt.linear_solver"] = "PARDISO"
    opts["ipopt.nlp_scaling_method"] = "gradient-based"
    opts["ipopt.derivative_test"] = "second-order"
    opts["ipopt.check_derivatives_for_naninf"] = "yes"
    opts["ipopt.tol"] = 1e-25
    # create solver
    dae = {'x': c.vertcat(mass_frac_out_CO2, temp),
           'p': para_sym,
           'g': c.vertcat(dwCO2dt, dmcpTdt)}
    # Create a implicit function instance to solve the system of equations
    # molar ratio H2 to CO2
    ratio_H2_CO2 = [4, 1]
    # ratio H2O to CH4
    ratio_H2O_CH4 = [2, 1]
    # number of experiments
    num = 10000
    # mole fraction
    mole_frac_in_vec = np.zeros((num, 5))
    # CO2
    mole_frac_in_vec[:, 1] = (1-portion_product_in_feed_vec) \
        * (ratio_H2_CO2[1]/sum(ratio_H2_CO2))
    # H2O
    mole_frac_in_vec[:, 2] = portion_product_in_feed_vec \
        * (ratio_H2O_CH4[0]/sum(ratio_H2O_CH4))
    # H2
    mole_frac_in_vec[:, 3] = (1-portion_product_in_feed_vec) \
        * (ratio_H2_CO2[0]/sum(ratio_H2_CO2))
    mole_frac_in_vec[:, 4] = 0  # N2
    rf = c.nlpsol('rf', 'ipopt', dae, opts)
    for mole_frac_number in range(len(mole_frac_in_vec)):
        arg = {}
        # Parameter
        temp_in = temp_in_vec[mole_frac_number]
        pressure = pressure_vec[mole_frac_number]
        time_residence = time_residence_vec[mole_frac_number]
        mole_frac_in = mole_frac_in_vec[mole_frac_number, :]
        portion_product_in_feed = portion_product_in_feed_vec[mole_frac_number]
        para = [temp_in, pressure, time_residence,
                mole_frac_in[0], mole_frac_in[1],
                mole_frac_in[2], mole_frac_in[3], mole_frac_in[4],
                portion_product_in_feed]
        arg["p"] = para
        # Initial condition
        initial_value_mass_frac_CO2_0 = 0.4
        initial_value_temp = temp_in+100
        initial_values = c.vertcat(initial_value_mass_frac_CO2_0,
                                   initial_value_temp)
        arg["x0"] = initial_values
        # Bounds on x
        arg["lbx"] = [0, temp_in]
        arg["ubx"] = [1, 1000]
        # Bounds on g
        arg["lbg"] = [0, 0]
        arg["ubg"] = [0, 0]
        rf_results = rf(**arg)
        if (rf_results["g"][0] < 1e-13 and rf_results["g"][1] < 1e-13):
            mass_frac_out_CO2_dae = rf_results["x"][0]
            temp_dae = rf_results["x"][1]

            # Calculate Conversion of CO2
            # Inlet Mass fraction of all Species
            mass_frac_in_vec = mole_frac_in*MOLMASS_VEC \
                / np.sum(mole_frac_in*MOLMASS_VEC)
            # sanity check
            assert np.isclose(np.sum(mass_frac_in_vec), 1, rtol=0.01)

            # Conversion of CO2
            mass_frac_in_CO2 = mass_frac_in_vec[0, 1]
            conv_CO2 = (mass_frac_in_CO2-mass_frac_out_CO2_dae) \
                / mass_frac_in_CO2
            # saving results
            temp_in_all = np.append(temp_in_all, temp_in)
            experiment_no = np.append(experiment_no, mole_frac_number)
            temp_all = np.append(temp_all, temp_dae)
            conv_CO2_all = np.append(conv_CO2_all, conv_CO2)

            # Inlet concentrations of all species
            conc_in = mole_frac_in*density_gas_in \
                / (np.sum([mole_frac_in * MOLMASS_VEC],
                          axis=2).flatten()[:, None])

            # Molar Concentration of all Species in mol m^-3
            conc_vec = concentrations_from_conversion(conc_in, conv_CO2,
                                                      stoic_vec).flatten()
            # Mole fraction of all Species
            mole_frac_vec = conc_vec/np.sum(conc_vec, axis=0)

            try:
                if mole_frac_number == 0:
                    conc_vec_all = conc_vec
                    mole_frac_vec_all = mole_frac_vec
                else:
                    conc_vec_all = np.vstack([conc_vec_all, conc_vec])
                    mole_frac_vec_all = np.vstack([mole_frac_vec_all,
                                                   mole_frac_vec])
            except Exception:
                conc_vec_all = conc_vec
                mole_frac_vec_all = mole_frac_vec
    experiment_no = list(experiment_no.astype(int))
    return conv_CO2_all, temp_all, experiment_no
