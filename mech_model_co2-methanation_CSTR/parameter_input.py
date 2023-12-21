"""
Created on Mon Feb 14 08:22:04 2022.

@author: peterson
"""
# extern
import numpy as np

##############################################################################
# Parameter Input
##############################################################################

'''Parameter input'''
# Inlet molar ratio H2 to CO2
ratio_H2_CO2_in = 4
# Molar inlet concentrations in mol m^-3
conc_reference = 40
conc_in_vec = np.zeros((1, 5))
conc_in_vec[0, 0] = 0.0  # CH4
conc_in_vec[0, 1] = conc_reference  # CO2
conc_in_vec[0, 2] = 0.0  # H2O
conc_in_vec[0, 3] = ratio_H2_CO2_in*conc_reference  # H2
conc_in_vec[0, 4] = conc_reference  # N2
# Inlet Temperature in K
temp_in = 600
# Inlet Pressure in bar
pressure_in = 5
# heat capacity in J kg^-1 K^-1
heat_cap_in = 2359
# Inet gas density in kg m^-3
density_gas_in = 0.5
# Inlet gas flow in m s^-1
velocity_gas_in = 1

# Stoichiometric matrix
stoic_vec = np.zeros((1, 5))
stoic_vec[0, 0] = +1.0  # CH4
stoic_vec[0, 1] = -1.0  # CO2
stoic_vec[0, 2] = +2.0  # H2O
stoic_vec[0, 3] = -4.0  # H2
stoic_vec[0, 4] = +0.0  # N2

'''Parameters input dictionary'''
input_params = dict()
input_params['conc_in_vec'] = conc_in_vec
input_params['temp_in'] = temp_in
input_params['pressure_in'] = pressure_in
input_params['heat_cap_in'] = heat_cap_in
input_params['density_gas_in'] = density_gas_in
input_params['velocity_gas_in'] = velocity_gas_in
input_params['stoic_vec'] = stoic_vec
# verify mass conservation
assert np.linalg.matrix_rank(stoic_vec) < stoic_vec.shape[1]

##############################################################################
# Parameter LHHW Equation
##############################################################################

'''Parameter LHHW rate equation'''
# Source: Koschany2016
# Reference Temperature in Kelvin
temp_ref = 555
# Pre-exponential factor in mol bar^-1 gcat^-1 s^-1
k0 = 3.46E-4
# Activation Energy in kJ mol^-1
energy_act = 77.5
# Eq. Konstant Hydrogen in bar^-0.5
eq_const_H2_0 = 0.44
# Eq. Constant Hydroxyl in bar^-0.5
eq_const_OH_0 = 0.5
# Eq. Constant Mix in bar^-0.5
eq_const_mix_0 = 0.88
# Enthalpy Difference Hydrogen in kJ mol^-1
dH_H2 = -6.2
# Enthalpy Difference Hydroxyl in kJ mol^-1
dH_OH = 22.4
# Enthalpy Difference Mix in kJ mol^-1
dH_mix = -10

'''LHHW dictionary'''
LHHW_params = dict()
LHHW_params['temp_ref'] = temp_ref
LHHW_params['k0'] = k0
LHHW_params['energy_act'] = energy_act
LHHW_params['eq_const_H2_0'] = eq_const_H2_0
LHHW_params['eq_const_OH_0'] = eq_const_OH_0
LHHW_params['eq_const_mix_0'] = eq_const_mix_0
LHHW_params['dH_H2'] = dH_H2
LHHW_params['dH_OH'] = dH_OH
LHHW_params['dH_mix'] = dH_mix

##############################################################################
# Parameter Power Law Equation
##############################################################################

'''Parameter Power Law'''
# Source: Koschany2016
# Reference Temperature in Kelvin
temp_ref = 555
# Pre-exponential factor in mol bar^-0.54 s^-1 gcat^-1
k0 = 6.41E-5
# Activation Energy in kJ mol^-1
energy_act = 93.6
# Exponent for Carbon Dioxide
n_CO2 = 0.16
# Exponent for Hydrogen
n_H2 = 0.31
#  Eq. Constant Hydroxyl in bar^-0.5
eq_const_OH_0 = 0.62
# Enthalpy Difference Hydroxyl in kJ mol^-1
dH_OH = 64.3

'''PL dictionary'''
PL_params = dict()
PL_params['temp_ref'] = temp_ref
PL_params['k0'] = k0
PL_params['energy_act'] = energy_act
PL_params['n_H2'] = n_H2
PL_params['n_CO2'] = n_CO2
PL_params['eq_const_OH_0'] = eq_const_OH_0
PL_params['dH_OH'] = dH_OH

##############################################################################
# Parameter Input
##############################################################################

'''Reactor Specifications'''
# Bremer2021_FER_supp
# Catalyst density (porous material) in g m^-3
density_cat = 1032*1e3
# Bed void fraction [-]
void_frac_bed = 0.39
# Effectiveness factor [-], Range between 0.05 and 1
factor_eff = 1
# Length of the reactor in m
# not given in Bremer2021_FER_supp
length_reactor = 0.5
# Radius of the reactor in m
radius_reactor = 0.02
# heat transfer coefficient in W m^-2 K^-1
coef_heat_transfer = 200
# Reactor Volume in m^3
volume_reactor = np.pi*radius_reactor**2*length_reactor


'''Reactor specification dictionary'''
reactor_params = dict()
reactor_params['density_cat'] = density_cat
reactor_params['void_frac_bed'] = void_frac_bed
reactor_params['factor_eff'] = factor_eff
reactor_params['length_reactor'] = length_reactor
reactor_params['radius_reactor'] = radius_reactor
reactor_params['volume_reactor'] = volume_reactor
reactor_params['coef_heat_transfer'] = coef_heat_transfer
