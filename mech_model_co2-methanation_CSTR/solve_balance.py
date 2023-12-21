"""
Created on Mon Mar 28 12:53:38 2022.

@author: peterson
"""
# functions

# extern
import casadi as c
import numpy as np

from balance import mass_energy_balance
from graphics import graphics_dae
# dictionariesequlibrium
from parameter_input import input_params
from utils import (concentrations_from_conversion,
                   constants,
                   optimize_equality_conversion)


def solve_DAE(temp_in_vec, pressure_vec,
              time_residence_vec, mole_frac_in_vec, equilibrium):
    """Solve DAE to get steady state data."""
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
    equilibrium_conversion_all = []
    # Solve DAE
    [mass_frac_out_CO2, temp, para_sym,
     dwCO2dt, dmcpTdt] = mass_energy_balance()
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
    rf = c.nlpsol('rf', 'ipopt', dae, opts)
    for mole_frac_number in range(len(mole_frac_in_vec)):
        arg = {}
        # Parameter
        temp_in = temp_in_vec[mole_frac_number]
        pressure = pressure_vec[mole_frac_number]
        velocity_gas_in = time_residence_vec[mole_frac_number]
        mole_frac_in = mole_frac_in_vec[mole_frac_number, :]
        para = [temp_in, pressure, velocity_gas_in,
                mole_frac_in[0], mole_frac_in[1],
                mole_frac_in[2], mole_frac_in[3], mole_frac_in[4]]
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
        if (abs(rf_results["g"][0]) < 1e-13 and abs(rf_results["g"][1]) < 1e-13):
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

            # get equilibrium conversion
            if equilibrium is True:
                equilibrium_conversion = optimize_equality_conversion(
                    temp_dae, pressure, conc_in, stoic_vec)
                equilibrium_conversion_all = np.append(
                    equilibrium_conversion_all, equilibrium_conversion)

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

        if len(temp_in_all) == 1000:
            break
    # Graphics
    graphics_dae(temp_all, conv_CO2_all, mole_frac_vec_all)
    experiment_no = list(experiment_no.astype(int))
    return conv_CO2_all, temp_all, experiment_no, equilibrium_conversion_all
