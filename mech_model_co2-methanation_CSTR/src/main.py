"""
Created on Mon Feb 14 08:16:02 2022.

@author: peterson
"""
import numpy as np

# dictionaries
from smt.sampling_methods import LHS
from solve_balance import solve_DAE
from utils import constants, generate_output_dae

if __name__ == "__main__":

    ###########################################################################
    # Flags
    ###########################################################################
    extrapolation_pressure = False
    extrapolation_temperature = False
    output = True
    equilibrium = False
    file_name = ('data')

    ###########################################################################
    # Latin Hypercube Sampling
    ###########################################################################
    # number of experiments
    num = 50000
    # Upper and lower bounds for pressure in bar
    pressure_bounds = [1, 10]
    # Upper and lower bounds for temperature in kelvin
    temperature_in_bounds = [400, 800]
    # temperature_in_bounds = [453, 613]
    # residence time in 1/s
    time_residence_bounds = [0.04, 0.4]
    # Upper and lower bounds for ratio between product and educt
    portion_product_in_feed_bounds = [0, 0.5]
    # take of 10 percent of upper and lower bounds to have extrapolation data
    if extrapolation_pressure is True:
        # pressure
        pressure_bounds = [10, 15]
    if extrapolation_temperature is True:
        # pressure
        temperature_in_bounds = [800, 900]
    xlimits = np.array([pressure_bounds,
                        temperature_in_bounds,
                        time_residence_bounds,
                        portion_product_in_feed_bounds])
    sampling = LHS(xlimits=xlimits, criterion='center', random_state=1)
    # call LHS
    x = sampling(num)
    # Unpack
    # Pressure in bar
    pressure_vec = x[:, 0]
    # Temperature in Kelvin
    temp_in_vec = x[:, 1]
    # inlet superficial velocity
    time_residence_vec = x[:, 2]
    # Residence time in the Reactor in s
    # reflux of products
    portion_product_in_feed = x[:, 3]
    # Build vector of concentrations
    mole_frac_in_vec = np.zeros((num, 5))
    # molar ratio H2 to CO2
    ratio_H2_CO2 = [4, 1]
    # ratio H2O to CH4
    ratio_H2O_CH4 = [2, 1]
    # CH4
    mole_frac_in_vec[:, 0] = portion_product_in_feed*(ratio_H2O_CH4[1]
                                                      / sum(ratio_H2O_CH4))
    # CO2
    mole_frac_in_vec[:, 1] = (1-portion_product_in_feed)*(ratio_H2_CO2[1]
                                                          / sum(ratio_H2_CO2))
    # H2O
    mole_frac_in_vec[:, 2] = portion_product_in_feed*(ratio_H2O_CH4[0]
                                                      / sum(ratio_H2O_CH4))
    # H2
    mole_frac_in_vec[:, 3] = (1-portion_product_in_feed)*(ratio_H2_CO2[0]
                                                          / sum(ratio_H2_CO2))
    mole_frac_in_vec[:, 4] = 0  # N2
    # Molar Mass in kg mol^-1
    MOLMASS_VEC = constants['MOLMASS_VEC']

    ###########################################################################
    # Solve DAE
    ###########################################################################
    conv_CO2, temp, experiment_no, equilibrium_conversion, \
        = solve_DAE(temp_in_vec, pressure_vec,
                    time_residence_vec, mole_frac_in_vec, equilibrium)

    ###########################################################################
    # Generate Output
    ###########################################################################
    if output is True:
        generate_output_dae(pressure_vec[experiment_no],
                            temp_in_vec[experiment_no],
                            time_residence_vec[experiment_no],
                            portion_product_in_feed[experiment_no],
                            conv_CO2, temp,
                            file_name)
