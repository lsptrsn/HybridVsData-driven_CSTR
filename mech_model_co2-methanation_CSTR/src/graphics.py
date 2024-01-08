"""
Created on Tue Feb 22 09:51:06 2022.

@author: peterson
"""
# extern
import matplotlib.pyplot as plt
import numpy as np


def graphics_dae(temp, conv_CO2, mole_frac_vec):
    """Display results for stationary case."""
    # Conversion over Temperature
    plt.plot(temp, conv_CO2, 'bx', label='X_CO2')
    plt.legend(loc='best')
    plt.xlabel('Temperature in K')
    plt.ylabel('Conversion of CO2')
    plt.ylim([0, 1])
    plt.grid()
    plt.show()

    # Mole Fractions over Temperature
    plt.plot(temp, mole_frac_vec[:, 0], 'bx', label='CH4')
    plt.plot(temp, mole_frac_vec[:, 1], 'rx', label='CO2')
    plt.plot(temp, mole_frac_vec[:, 2], 'gx', label='H2O')
    plt.plot(temp, mole_frac_vec[:, 3], 'cx', label='H2')
    plt.legend(loc='best')
    plt.xlabel('Temperature')
    plt.ylabel('Mole Fractions of the Species')
    plt.grid()
    plt.show()
    return
