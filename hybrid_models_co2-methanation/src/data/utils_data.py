"""Utility functions."""
from denseweight import DenseWeight
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import os
import pandas as pd
from scipy import constants
from sklearn.model_selection import train_test_split
import sys

import paths
from mechanistic_part.mechanistics import (get_partial_pressure,
                                           get_conversion)
from mechanistic_part.reaction_rate import power_law


def create_dataframe_for_serial_modeling(df, filename):
    """With mechanistic part, create a dataframe for hybrid modeling."""
    partial_pressure = get_partial_pressure(df['pressure in bar'],
                                            df['portion of product in feed'],
                                            df['conversion CO2'])
    df = df.reset_index(drop=True)
    partial_pressure = partial_pressure.reset_index(drop=True)
    # Remove physically impossible values
    df[df < 0] = 0
    partial_pressure[partial_pressure < 0] = 0
    # equilibrium term
    # Universal gas constant in kJ mol^-1 K^-1
    RG = constants.R*1e-3
    # Approximated by an empirical formula (Koschany2016 Eq. 30)
    equilibrium_const = 137*df['temperature in K']**(-3.998) \
        * np.exp(158.7/RG/df['temperature in K'])
    # Equilibrium Constant in bar^-2
    equilibrium_const = equilibrium_const*1.01325**-2
    quotient = partial_pressure.iloc[:, 0]*partial_pressure.iloc[:, 2]**2 \
        / (partial_pressure.iloc[:, 1]*partial_pressure.iloc[:, 3]**4
           * equilibrium_const)
    potential_term = 1-quotient
    # build dict of hybrid modeling params
    dict_serial = {'partial pressure CH4 in bar': partial_pressure.iloc[:, 0],
                   'partial pressure CO2 in bar': partial_pressure.iloc[:, 1],
                   'partial pressure H2O in bar': partial_pressure.iloc[:, 2],
                   'partial pressure H2 in bar': partial_pressure.iloc[:, 3],
                   'inlet temperature in K': df['inlet temperature in K'],
                   'temperature in K': df['temperature in K'],
                   'residence time in s': df['residence time in s'],
                   'portion of product in feed': df['portion of product in feed'],
                   # 'potential_term': potential_term,
                   'conversion CO2': df['conversion CO2']}
    df_serial = pd.DataFrame(dict_serial)
    if filename is not None:
        filename_serial = 'serial_' + filename
        df_serial.to_csv(paths.interim_data_path / filename_serial,
                         index=False)
    return df_serial


def create_dataframe_for_parallel_modeling(df, PL_params, filename=None,
                                            evaluate=False):
    """With mechanistic part, create dataframe for parallel hybrid modeling."""
    partial_pressure = get_partial_pressure(df['pressure in bar'],
                                            df['portion of product in feed'],
                                            df['conversion CO2'])
    reaction_rate_PL = power_law(PL_params,
                                  df['temperature in K'],
                                  partial_pressure.iloc[:, 0],
                                  partial_pressure.iloc[:, 1],
                                  partial_pressure.iloc[:, 2],
                                  partial_pressure.iloc[:, 3])
    conv_CO2_PL = get_conversion(df['residence time in s'],
                                  df['portion of product in feed'],
                                  reaction_rate_PL)
    residuals_PL = conv_CO2_PL - df.iloc[:, -1]
    df_parallel = {'partial pressure CH4 in bar': partial_pressure.iloc[:, 0],
                    'partial pressure CO2 in bar': partial_pressure.iloc[:, 1],
                    'partial pressure H2O in bar': partial_pressure.iloc[:, 2],
                    'partial pressure H2 in bar': partial_pressure.iloc[:, 3],
                    'inlet temperature in K': df['inlet temperature in K'],
                    'temperature in K': df['temperature in K'],
                    'residence time in s': df['residence time in s'],
                    'portion of product in feed': df[
                        'portion of product in feed'],
                    'pl residuals': residuals_PL}
    if evaluate is True:
        # addig conversion to dataframe
        df_parallel = {'partial pressure CH4 in bar': partial_pressure.iloc[:,
                                                                            0],
                        'partial pressure CO2 in bar': partial_pressure.iloc[:,
                                                                            1],
                        'partial pressure H2O in bar': partial_pressure.iloc[:,
                                                                            2],
                        'partial pressure H2 in bar': partial_pressure.iloc[:,
                                                                            3],
                        'inlet temperature in K': df['inlet temperature in K'],
                        'temperature in K': df['temperature in K'],
                        'residence time in s': df['residence time in s'],
                        'portion of product in feed': df[
                            'portion of product in feed'],
                        'pl residuals': residuals_PL,
                        'conversion CO2':  df['conversion CO2']}
    df_parallel = pd.DataFrame(df_parallel)
    if filename is not None:
        filename_parallel = 'parallel_' + filename
        df_parallel.to_csv(paths.interim_data_path / filename_parallel,
                            index=False)
    return df_parallel, conv_CO2_PL


def create_dataframe_for_PL_modeling(df, PL_params, filename=None,
                                            evaluate=False):
    """With mechanistic part, create dataframe for parallel hybrid modeling."""
    partial_pressure = get_partial_pressure(df['pressure in bar'],
                                            df['portion of product in feed'],
                                            df['conversion CO2'])
    reaction_rate_PL = power_law(PL_params,
                                 df['temperature in K'],
                                 partial_pressure.iloc[:, 0],
                                 partial_pressure.iloc[:, 1],
                                 partial_pressure.iloc[:, 2],
                                 partial_pressure.iloc[:, 3])
    conv_CO2_PL = get_conversion(df['residence time in s'],
                                  df['portion of product in feed'],
                                  reaction_rate_PL)
    residuals_PL = conv_CO2_PL - df.iloc[:, -1]
    df_parallel = {'partial pressure CH4 in bar': partial_pressure.iloc[:, 0],
                    'partial pressure CO2 in bar': partial_pressure.iloc[:, 1],
                    'partial pressure H2O in bar': partial_pressure.iloc[:, 2],
                    'partial pressure H2 in bar': partial_pressure.iloc[:, 3],
                    'inlet temperature in K': df['inlet temperature in K'],
                    'temperature in K': df['temperature in K'],
                    'residence time in s': df['residence time in s'],
                    'portion of product in feed': df[
                        'portion of product in feed'],
                    'pl residuals': residuals_PL}
    if evaluate is True:
        # addig conversion to dataframe
        df_parallel = {'partial pressure CH4 in bar': partial_pressure.iloc[:,
                                                                            0],
                        'partial pressure CO2 in bar': partial_pressure.iloc[:,
                                                                            1],
                        'partial pressure H2O in bar': partial_pressure.iloc[:,
                                                                            2],
                        'partial pressure H2 in bar': partial_pressure.iloc[:,
                                                                            3],
                        'inlet temperature in K': df['inlet temperature in K'],
                        'temperature in K': df['temperature in K'],
                        'residence time in s': df['residence time in s'],
                        'portion of product in feed': df[
                            'portion of product in feed'],
                        'pl residuals': residuals_PL,
                        'conversion CO2':  df['conversion CO2']}
    df_parallel = pd.DataFrame(df_parallel)
    if filename is not None:
        filename_parallel = 'parallel_' + filename
        df_parallel.to_csv(paths.interim_data_path / filename_parallel,
                            index=False)
    return df_parallel, conv_CO2_PL


def split_train_test(df, filename, save=False, random_state=0):
    """Split data into training and testset."""
    filename_split = filename.split('.')[0]
    filename_test = filename_split+'_test.txt'
    try:
        df_test = pd.read_csv(paths.raw_data_path / 'test' / filename_test)
        df_train = df
    except FileNotFoundError:
        print('no external test set')
        df_train, df_test = train_test_split(df, test_size=100,
                                             shuffle=True,
                                             random_state=0)
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
    # save training and test
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    if save is True:
        # create target
        targetpath = parentdir+'/data/processed/'+filename.replace(".txt", "")
        # create target file if it does not exist yet
        if not os.path.exists(targetpath):
            os.makedirs(targetpath)
        # targetdir = os.path.join(parentdir, targetpath)
        sys.path.append(targetpath)
        # save data
        try:
            joblib.dump(df_train,
                        ('../data/processed/'
                         + filename.replace(".txt", "")
                         + '/df_train.pkl'))
            joblib.dump(df_test,
                        ('../data/processed/'
                         + filename.replace(".txt", "")
                         + '/df_test.pkl'))
        except FileNotFoundError:
            print('LHHW-not saved')
    return df_train, df_test


def split_train_val(df):
    """Split data into input and targets, into training and validationset."""
    df_train, df_val = train_test_split(df, test_size=100, shuffle=True,
                                        random_state=0)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    return df_train, df_val


def save_min_max(df):
    """Save minimal and maximal values from unscaled dataframe."""
    min_max_values = {
        'p CH4 max': np.max(df["partial pressure CH4 in bar"]),
        'p CH4 min': np.min(df["partial pressure CH4 in bar"]),
        'p CO2 max': np.max(df["partial pressure CO2 in bar"]),
        'p CO2 min': np.min(df["partial pressure CO2 in bar"]),
        'p H2O max': np.max(df["partial pressure H2O in bar"]),
        'p H2O min': np.min(df["partial pressure H2O in bar"]),
        'p H2 max': np.max(df["partial pressure H2 in bar"]),
        'p H2 min': np.min(df["partial pressure H2 in bar"]),
        'temp in max': np.max(df["inlet temperature in K"]),
        'temp in min': np.min(df["inlet temperature in K"]),
        'temp max': np.max(df["temperature in K"]),
        'temp min': np.min(df["temperature in K"]),
        'time residence max': np.max(df["residence time in s"]),
        'time residence min': np.min(df["residence time in s"]),
        'portion of product in feed max': np.max(df[
            "portion of product in feed"]),
        'portion of product in feed min': np.min(df[
            "portion of product in feed"])
    }
    try:
        joblib.dump(min_max_values, 'data/min_max_values.pkl')
    except FileNotFoundError:
        joblib.dump(min_max_values, '../data/min_max_values.pkl')
    return


def unscale_data(X):
    """Unscale Data with the helf of minimal and maximal values."""
    try:
        min_max_values = joblib.load('data/min_max_values.pkl')
    except FileNotFoundError:
        min_max_values = joblib.load('../data/min_max_values.pkl')
    # unscale partial pressure
    partial_pressure_CH4_scaled = X[:, 0]
    partial_pressure_CH4 = partial_pressure_CH4_scaled \
        * (min_max_values['p CH4 max']-min_max_values['p CH4 min']) \
        + min_max_values['p CH4 min']
    partial_pressure_CO2_scaled = X[:, 1]
    partial_pressure_CO2 = partial_pressure_CO2_scaled \
        * (min_max_values['p CO2 max']-min_max_values['p CO2 min']) \
        + min_max_values['p CO2 min']
    partial_pressure_H2O_scaled = X[:, 2]
    partial_pressure_H2O = partial_pressure_H2O_scaled \
        * (min_max_values['p H2O max']-min_max_values['p H2O min']) \
        + min_max_values['p H2O min']
    partial_pressure_H2_scaled = X[:, 3]
    partial_pressure_H2 = partial_pressure_H2_scaled \
        * (min_max_values['p H2 max']-min_max_values['p H2 min']) \
        + min_max_values['p H2 min']
    # unscale inlet temperature
    temperature_in_scaled = X[:, 4]
    temperature_in = temperature_in_scaled \
        * (min_max_values['temp in max']-min_max_values['temp in min']) \
        + min_max_values['temp in min']
    # unscale temperature
    temperature_scaled = X[:, 5]
    temperature = temperature_scaled \
        * (min_max_values['temp max']-min_max_values['temp min']) \
        + min_max_values['temp min']
    # unscale residence time
    time_residence_scaled = X[:, 6]
    time_residence = time_residence_scaled \
        * (min_max_values['time residence max']
            - min_max_values['time residence min']) \
        + min_max_values['time residence min']
    # unscale portion product in feed
    portion_product_in_feed_bounds_scaled = X[:, 7]
    portion_product_in_feed = portion_product_in_feed_bounds_scaled\
        * (min_max_values['portion of product in feed max']
            - min_max_values['portion of product in feed min']) \
        + min_max_values['portion of product in feed min']
    return (partial_pressure_CH4, partial_pressure_CO2, partial_pressure_H2O,
            partial_pressure_H2, temperature_in, temperature, time_residence,
            portion_product_in_feed)


def get_sample_weights(df_train, ml_type):
    """Get weights from Parameter distribution."""
    # https://github.com/SteiMi/denseweight
    dw = DenseWeight(alpha=1)
    if ml_type == 'datadriven':
        scaler = MinMaxScaler()
        df_train_scaled = scaler.fit_transform(df_train)
        # CO2 conversion
        sample_weights = dw.fit(np.array(df_train_scaled[:, -1]))
    elif ml_type == 'parallel':
        scaler = MinMaxScaler()
        df_train_scaled = scaler.fit_transform(df_train)
        # CO2 conversion
        sample_weights = dw.fit(np.array(df_train_scaled[:, -1]))
    else:
        # CO2 conversion
        sample_weights = dw.fit(np.array(df_train['conversion CO2']))
    return dw
