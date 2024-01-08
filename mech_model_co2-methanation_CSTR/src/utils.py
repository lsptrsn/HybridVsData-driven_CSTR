"""
Created on Mon Feb 14 09:48:23 2022.

@author: peterson
"""
# extern
from molmass import Formula
import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import minimize

# machine epsilon
eps = np.finfo(float).eps

# Universal gas constant in kJ mol^-1 K^-1
RG = constants.R*1e-3

# Molar Mass of the Species in kg mol^-1
MOLMASS_VEC = np.zeros((1, 5))
MOLMASS_VEC[0, 0] = Formula('CH4').mass*1e-3  # CH4
MOLMASS_VEC[0, 1] = Formula('CO2').mass*1e-3  # CO2
MOLMASS_VEC[0, 2] = Formula('H2O').mass*1e-3  # H2O
MOLMASS_VEC[0, 3] = Formula('H2').mass*1e-3  # H2
MOLMASS_VEC[0, 4] = Formula('N2').mass*1e-3  # N2

# Enthalpy Difference Methane in J mol^-1
DH_METH = -164.9*1000
# Gibbs Free Energy Change in KJ mol^-1
DG_METH = -113.635

constants = dict()
constants['RG'] = RG
constants['MOLMASS_VEC'] = MOLMASS_VEC
constants['DH_METH'] = DH_METH


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


def generate_output_dae(pressure, temperature_in, time_residence,
                        portion_product_in_feed, conv_CO2_vec, temperature,
                        filename):
    """Generate output files for DAE."""
    # Numpy to Dataframe and round to 5 descimals
    conv_CO2_vec = pd.DataFrame(conv_CO2_vec).round(5)
    conditions = np.hstack([pressure.reshape(-1, 1),
                            temperature_in.reshape(-1, 1),
                            temperature.reshape(-1, 1),
                            time_residence.reshape(-1, 1),
                            portion_product_in_feed.reshape(-1, 1)])
    conditions_out = pd.DataFrame(conditions).round(5)

    # concat condiions and results to one DataFrame
    out_df = pd.concat([conditions_out, conv_CO2_vec], axis=1)
    # CH4/CO2 H2O/CO2, H2/CO2 N2/CO2
    out_df.columns = ['pressure in bar', 'inlet temperature in K',
                      'temperature in K',
                      'residence time in s',
                      'portion of product in feed', 'conversion CO2']
    # Print to a .tx file
    out_df.to_csv(filename, index=None)


def estimate_equality_constant_via_vant_hoff(temperature, pressure):
    """Estimate concentrational equality constant by van't Hoff Equation."""
    RG = constants['RG']
    DG_METH = -113.635
    Keq = np.exp(-DG_METH/RG/temperature)
    Kc = Keq * (pressure/RG/temperature)**2
    return Kc


def estimate_equality_constant_via_mass_action_law(conc_in_vec,
                                                   conv_CO2,
                                                   stoic_vec):
    """Estimate concentrational equality constant by mass action law."""
    conc_vec = concentrations_from_conversion(conc_in_vec, conv_CO2, stoic_vec)
    Kc = conc_vec[0] * conc_vec[2]**2 / (conc_vec[1] * conc_vec[3]**4 + eps)
    return Kc


def objective_equality_conversion(conv_CO2,
                                  temperature,
                                  pressure,
                                  conc_in_vec,
                                  stoic_vec):
    """Objective function to be solved by the optimizer."""
    Kc_vant_hoff = estimate_equality_constant_via_vant_hoff(temperature,
                                                            pressure)
    Kc_mass_action_law = estimate_equality_constant_via_mass_action_law(
        conc_in_vec, conv_CO2, stoic_vec)
    objective = (Kc_vant_hoff - Kc_mass_action_law)**2
    return objective


def optimize_equality_conversion(temperature,
                                 pressure,
                                 conc_in_vec,
                                 stoic_vec):
    """Get equality conversion by equating equations for equality constants."""
    bnds = [(0, 1)]
    res = minimize(objective_equality_conversion,
                   x0=0.9,
                   args=(temperature, pressure, conc_in_vec, stoic_vec),
                   method='Nelder-Mead',
                   bounds=bnds)
    equality_conversion = res.x
    return equality_conversion
