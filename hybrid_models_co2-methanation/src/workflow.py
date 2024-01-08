"""Methods for joining up parts of a Reproducible Data Science workflow."""
import pandas as pd
import paths

from mechanistic_part.reaction_rate import fit_params_PL
from data.preprocessing import preprocess_data
from data.utils_data import (create_dataframe_for_serial_modeling,
                             create_dataframe_for_parallel_modeling,
                             create_dataframe_for_PL_modeling,
                             save_min_max,
                             split_train_test, split_train_val)
from models.train_model import (MLPR_spotcheck,
                                hyperparameter_optimization)
from models.score_benchmark import get_baseline_model
from postprocessing.visualize_results import run_visualize


def fitting_routine(df_train, df_test, name_addition, ml_type, PL_params=None):
    """Run machine learning part."""
    score_spotcheck = []
    score_baseline = []
    score = []
    # Get best possible and baseline score for comparison
    score_baseline = get_baseline_model(df_train, df_test)
    # Fit Model to data
    score_spotcheck = MLPR_spotcheck(df_train, df_test, ml_type)
    # result, model, score = hyperparameter_optimization(
    #     df_train, df_test, ml_type, name_addition
    #     )
    # run_visualize(result, ml_type, name_addition)
    return (score_spotcheck, score_baseline, score)


if __name__ == "__main__":
    ###########################################################################
    # Settings
    ###########################################################################
    name_addition = '_test'
    # machine learning types that should be run
    run_datadriven = True
    run_hybrid_serial = False
    run_hybrid_serial_equilibrium = False
    run_hybrid_parallel = False
    ###########################################################################
    # Get Data
    ###########################################################################
    # load interim mass balance data
    filename = 'data.txt'
    df_raw = pd.read_csv(paths.raw_data_path / filename)
    # Preprocess data
    df = preprocess_data(df_raw, filename)
    ###########################################################################
    # Datadriven
    ###########################################################################
    if run_datadriven is True:
        # Fit Datadriven Model to Data
        df_train_all, df_test = split_train_test(df, filename)
        df_train, df_val = split_train_val(df_train_all)
        if 'LHHW' not in filename:
            try:
                df_train = df_train.drop(['temperature in K'], axis=1)
                df_train[df_train <= 0] = 1e-10
                df_val = df_val.drop(['temperature in K'], axis=1)
                df_val[df_val <= 0] = 1e-10
            except KeyError:
                pass
        (
            scores_spotcheck_datadriven,
            score_baseline_datadriven,
            scores_datadriven
        ) = fitting_routine(
            df_train, df_val, name_addition, ml_type='datadriven'
        )

    ###########################################################################
    # Hybrid Serial without Equilibrium
    ###########################################################################
    if run_hybrid_serial is True:
        df_train_all, df_test = split_train_test(df, filename, save=True)
        df_train, df_val = split_train_val(df_train_all)
        # Get input data for the machine learning part of hybrid models
        df_train_serial = create_dataframe_for_serial_modeling(
            df_train, filename
        )
        df_val_serial = create_dataframe_for_serial_modeling(
            df_val, None
        )
        # save minimal and maximal values of dataframe for serial modeling
        save_min_max(df_train_serial)
        # Fit Hybrid Serial Model to Data
        (
            scores_spotcheck_serial,
            score_baseline_serial,
            scores_serial
        ) = fitting_routine(
            df_train_serial, df_val_serial, name_addition, ml_type='serial'
        )

    ###########################################################################
    # Hybrid Serial with Equlibrium
    ###########################################################################
    if run_hybrid_serial_equilibrium is True:
        df_train_all, df_test = split_train_test(df, filename, save=True)
        df_train, df_val = split_train_val(df_train_all)
        # Get input data for the machine learning part of hybrid models
        df_train_serial = create_dataframe_for_serial_modeling(
            df_train, filename
        )
        df_val_serial = create_dataframe_for_serial_modeling(
            df_val, None
        )
        # save minimal and maximal values of dataframe for serial modeling
        save_min_max(df_train_serial)
        # Fit Hybrid Serial Model to Data
        (
            scores_spotcheck_serial_eq,
            score_baseline_serial_eq,
            scores_serial_eq
        ) = fitting_routine(
            df_train_serial, df_val_serial, name_addition,
            ml_type='serial_equilibrium'
        )

    ###########################################################################
    # Hybrid Parallel
    ###########################################################################
    if run_hybrid_parallel is True:
        # Fit Powerlaw parameter
        # Split data into trainings and test set
        df_train_all, df_test = split_train_test(df, filename, save=True)
        df_train, df_val = split_train_val(df_train_all)
        results_PL = fit_params_PL(df_train)
        # Create Dataframe
        df_train_parallel, conv_CO2_PL = create_dataframe_for_PL_modeling(
            df_train, results_PL.x, filename)
        df_val_parallel, conv_CO2_PL = create_dataframe_for_PL_modeling(
            df_val, results_PL.x, None)
        # Fit Hybrid Parallel Model to Data
        (
            scores_spotcheck_parallel,
            score_baseline_parallel,
            scores_parallel
        ) = fitting_routine(
            df_train_parallel, df_val_parallel, name_addition, 'parallel',
            results_PL.x
        )


# to save session
# date = datetime.now().strftime("%Y%m%d")
# filename = date+'_model_results.pkl'
# dill.dump_session(filename)
# dill.load_session('20221124_hybrid_model_results.pkl')
