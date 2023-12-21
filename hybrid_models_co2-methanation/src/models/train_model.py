"""Training Machine Learning Models to Training Data."""
from functools import partial
import joblib
import numpy as np
import pandas as pd
import random
from sklearn.compose import TransformedTargetRegressor
# from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Integer, Real
import skorch
from skorch.callbacks import (Checkpoint,
                              TrainEndCheckpoint)
import sys
import torch
import warnings

from data.utils_data import get_sample_weights
from mechanistic_part.mechanistics import get_conversion, get_potential_term
from models.MLPR_architecture import (MLPR,
                                      MyNeuralNetRegressor,
                                      MyHybridLossFunction,
                                      MyMSELossFunction,
                                      MyHybridEquillibriumLossFunction,
                                      my_loss)

warnings.simplefilter(action='ignore', category=FutureWarning)
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
eps = sys.float_info.epsilon


def call_metrics(y_train, y_train_model, y_test, y_test_model):
    """Calculate metrics from model results."""
    MSE_train = mean_squared_error(y_train, y_train_model)
    MAPE_train = mean_absolute_percentage_error(y_train, y_train_model)
    R2_train = r2_score(y_train, y_train_model)
    MSE_test = mean_squared_error(y_test, y_test_model)
    MAPE_test = mean_absolute_percentage_error(y_test, y_test_model)
    R2_test = r2_score(y_test, y_test_model)
    score = ({'MSE Training': MSE_train,
              'MAPE Train': MAPE_train,
              'R2 Training': R2_train,
              'MSE Validation': MSE_test,
              'MAPE Validation': MAPE_test,
              'R2 Validation': R2_test})
    score = pd.DataFrame(score, index=[0])
    return score


def call_regressor(df_train, df_test, ml_type):
    """Call Regressor."""
    X_train = df_train.iloc[:, 0:-1].values
    # get sample weights
    dw_initialized = get_sample_weights(df_train, ml_type)
    dirname = 'fit_results/fit_results_'+ml_type
    cp = Checkpoint(dirname=dirname, monitor='valid_loss_best')
    clipping = skorch.callbacks.GradientNormClipping(1)
    train_end_cp = TrainEndCheckpoint(dirname=dirname)
    if ml_type == 'serial':
        criterion = MyHybridLossFunction(dw=dw_initialized)
    elif ml_type == 'serial_equilibrium':
        criterion = MyHybridEquillibriumLossFunction(dw=dw_initialized)
    else:
        criterion = MyMSELossFunction(dw=dw_initialized, ml_type=ml_type)
    regressor = MyNeuralNetRegressor(
        module=MLPR(feature_count=X_train.shape[1]),
        # Reference to Loss Funcion
        criterion=criterion,
        # criterion__weight=weight
        # SGD, Adam, Autograd, ...
        optimizer=torch.optim.NAdam,
        optimizer__lr=0.005,
        # optimizer__weight_decay=1e-6,
        max_epochs=5000,
        batch_size=32,
        train_split=skorch.dataset.ValidSplit(cv=5),
        # train_split=None,
        callbacks=[cp, clipping, train_end_cp],
        warm_start=False,
        verbose=0,
        iterator_train__shuffle=True,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    return regressor


def MLPR_spotcheck(df_train, df_test, ml_type):
    """Train different regression method to training data."""
    X_train = df_train.iloc[:, 0:-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, 0:-1].values
    y_test = df_test.iloc[:, -1].values
    regressor = call_regressor(df_train, df_test, ml_type)
    pipe_contents = [('Scaler', MinMaxScaler(clip=True)),
                     ('algorithm', regressor)]
    pipe = Pipeline(pipe_contents)
    if ml_type == 'serial':
        model = TransformedTargetRegressor(regressor=pipe,
                                           transformer=MinMaxScaler(
                                               feature_range=(0, 0.5)))
    elif ml_type == 'serial_equilibrium':
        model = TransformedTargetRegressor(regressor=pipe,
                                           transformer=MinMaxScaler(
                                               feature_range=(0, 0.1)))
    else:
        model = TransformedTargetRegressor(regressor=pipe,
                                           transformer=MinMaxScaler())
    torch.manual_seed(0)
    model.regressor['algorithm'].initialize()
    model.fit(X_train, y_train)
    y_train_model = model.predict(X_train)
    y_train_model = np.clip(y_train_model, 0, 1)
    y_test_model = model.predict(X_test)
    if ml_type == 'serial':
        reaction_rate_train = y_train_model.reshape(-1, 1)
        y_train_model = get_conversion(X_train[:, 6], X_train[:, 7],
                                       reaction_rate_train.flatten(),
                                       length_reactor=None)
        reaction_rate_test = y_test_model.reshape(-1, 1)
        y_test_model = get_conversion(X_test[:, 6], X_test[:, 7],
                                      reaction_rate_test.flatten(),
                                      length_reactor=None)
    elif ml_type == 'serial_equilibrium':
        potential_term = get_potential_term(X_train).reshape(-1, 1)
        reaction_rate_train = y_train_model.reshape(-1, 1)*potential_term
        y_train_model = get_conversion(X_train[:, 6], X_train[:, 7],
                                       reaction_rate_train.flatten(),
                                       length_reactor=None)
        potential_term = get_potential_term(X_test).reshape(-1, 1)
        reaction_rate_test = y_test_model.reshape(-1, 1)*potential_term
        y_test_model = get_conversion(X_test[:, 6], X_test[:, 7],
                                      reaction_rate_test.flatten(),
                                      length_reactor=None)
    y_test_model = np.clip(y_test_model, 0, 1)
    score = call_metrics(y_train, y_train_model, y_test, y_test_model)
    return score


def weight_reset(m):
    """Reset Parameter weights."""
    if isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def optimize(params, param_names, pipe, df_train, df_test, ml_type):
    """Optimization function with arguments from search space."""
    # load data
    X_train = df_train.iloc[:, 0:-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, 0:-1].values
    y_test = df_test.iloc[:, -1].values
    # convert params to dictionary
    params = dict(zip(param_names, params))
    params['regressor__algorithm__module__n_neurons_per_layer'] = \
        int(params['regressor__algorithm__module__n_neurons_per_layer'])
    params['regressor__algorithm__batch_size'] = \
        int(params['regressor__algorithm__batch_size'])
    # initialize model with current parameters
    print(params)
    model = pipe.set_params(**params)
    model.regressor['algorithm'].module__feature_count = X_train.shape[1]
    loss_all = []
    for seed in range(10):
        model.regressor['algorithm'].module.model_stack.apply(weight_reset)
        torch.manual_seed(seed)
        model.fit(X_train, y_train)
        y_test_predict = model.predict(X_test)
        if ml_type == "serial" or ml_type == "serial_equilibrium":
            mse_seed = my_loss(y_test, y_test_predict, ml_type, X=X_test)
        else:
            mse_seed = mean_squared_error(y_test, y_test_predict)
        loss_seed = mse_seed
        loss_all.append(loss_seed)
        print(loss_all)
    loss = np.mean(loss_all)
    return loss


def optimize_hyperparameter(regressor, param_space, param_names,
                            df_train, df_test, default_parameter, ml_type):
    """Framework to train models with hyperparameter fitting."""
    # get pipe
    pipe_contents = [('scaler', MinMaxScaler()),
                     ('algorithm', regressor)]
    pipe = Pipeline(pipe_contents)
    if ml_type == 'serial':
        pipe_tt = TransformedTargetRegressor(regressor=pipe,
                                             transformer=MinMaxScaler(
                                                 feature_range=(0, 0.5)))
    elif ml_type == 'serial_equilibrium':
        pipe_tt = TransformedTargetRegressor(regressor=pipe,
                                             transformer=MinMaxScaler(
                                                 feature_range=(0, 0.1)))
    else:
        pipe_tt = TransformedTargetRegressor(regressor=pipe,
                                             transformer=MinMaxScaler())
    # call optimization function with partial
    optimization_function = partial(
        optimize,
        param_names=param_names,
        pipe=pipe_tt,
        df_train=df_train,
        df_test=df_test,
        ml_type=ml_type)
    # call gp_minimize from skopt
    result = gp_minimize(
        # Function to minimize
        func=optimization_function,
        # List of search space dimensions
        dimensions=param_space,
        # Number of calls to func
        n_calls=50,
        # Number of evaluations of func with initialization points
        n_initial_points=10,
        # The number of restarts of the optimizer when acq_optimizer is "lbfgs"
        n_restarts_optimizer=5,
        # initial values
        x0=default_parameter,
        # Set random state for reproducible results
        random_state=0,
        # Control the verbosity
        verbose=True,
        # Number of cores to run in parallel
        n_jobs=-1)
    # create best params dict and print it
    best_params = dict(zip(param_names, result.x))
    best_params['regressor__algorithm__module__n_neurons_per_layer'] = \
        int(best_params['regressor__algorithm__module__n_neurons_per_layer'])
    best_params['regressor__algorithm__batch_size'] = \
        int(best_params['regressor__algorithm__batch_size'])
    print(best_params)
    model = pipe_tt.set_params(**best_params)
    # load data
    X_train = df_train.iloc[:, 0:-1].values
    y_train = df_train.iloc[:, -1].values
    model.fit(X_train, y_train)
    return result, model


def get_scores(model, df_train, df_test, ml_type):
    """Get scores for training and validation."""
    # split data
    X_train = df_train.iloc[:, 0:-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, 0:-1].values
    y_test = df_test.iloc[:, -1].values
    if ml_type == 'serial':
        model.fit(X_train, y_train)
        y_train_model = model.predict(X_train)
        # y_train_model = np.exp(y_train_model)
        reaction_rate_train = y_train_model.reshape(-1, 1)
        y_train_model = get_conversion(X_train[:, 6], X_train[:, 7],
                                       reaction_rate_train.flatten(),
                                       length_reactor=None)
    elif ml_type == 'serial_equilibrium':
        model.fit(X_train, y_train)
        y_train_model = model.predict(X_train)
        potential_term = get_potential_term(X_train).reshape(-1, 1)
        reaction_rate_train = y_train_model.reshape(-1, 1)*potential_term
        y_train_model = get_conversion(X_train[:, 6], X_train[:, 7],
                                       reaction_rate_train.flatten(),
                                       length_reactor=None)
    else:
        model.fit(X_train, y_train)
        y_train_model = model.predict(X_train)
    y_test_model = model.predict(X_test)
    if ml_type == 'serial':
        reaction_rate_test = y_test_model.reshape(-1, 1)
        y_test_model = get_conversion(X_test[:, 6], X_test[:, 7],
                                      reaction_rate_test.flatten(),
                                      length_reactor=None)
    elif ml_type == 'serial_equilibrium':
        potential_term = get_potential_term(X_test).reshape(-1, 1)
        reaction_rate_test = y_test_model.reshape(-1, 1)*potential_term
        y_test_model = get_conversion(X_test[:, 6], X_test[:, 7],
                                      reaction_rate_test.flatten(),
                                      length_reactor=None)
    y_train_model = np.clip(y_train_model, 0, 1)
    y_test_model = np.clip(y_test_model, 0, 1)
    score = call_metrics(y_train, y_train_model, y_test, y_test_model)
    return score


def hyperparameter_optimization(df_train, df_test, ml_type, name_addition=''):
    """Train Data via Multilayer Perceptron regression."""
    # regressor
    regressor = call_regressor(df_train, df_test, ml_type)
    # Define param space
    param_space = [
        # Integer(1, 3, name='regressor__algorithm__module__n_hidden_layer'),
        Integer(5,  30, 'uniform',
                name='regressor__algorithm__module__n_neurons_per_layer'),
        Real(1e-3, 1e-2, 'log-uniform',
             name='regressor__algorithm__optimizer__lr'),
        Integer(16, 64, name='regressor__algorithm__batch_size')
    ]
    # Make a list of param names (same order as search space)
    param_names = [
        # 'regressor__algorithm__module__n_hidden_layer',
        'regressor__algorithm__module__n_neurons_per_layer',
        'regressor__algorithm__optimizer__lr',
        'regressor__algorithm__batch_size'
        ]
    # Scikit Learn default parameter
    default_parameter = [15, 5e-3, 32]
    # run hyper-parameter optimization
    result, model = optimize_hyperparameter(regressor,
                                            param_space,
                                            param_names,
                                            df_train,
                                            df_test,
                                            default_parameter,
                                            ml_type)
    score = get_scores(model, df_train, df_test, ml_type)
    model.regressor['algorithm'].initialize()
    if ml_type == 'serial':
        model.regressor['algorithm'].set_params(
            criterion=MyHybridLossFunction(dw=None))
    elif ml_type == 'serial_equilibrium':
        model.regressor['algorithm'].set_params(
            criterion=MyHybridEquillibriumLossFunction(dw=None))
    else:
        model.regressor['algorithm'].set_params(
            criterion=MyMSELossFunction(dw=None))
    X_train = df_train.iloc[:, 0:-1].values
    y_train = df_train.iloc[:, -1].values
    model.fit(X_train, y_train)
    result_name = '../models/' + ml_type + '_result'+name_addition+'.pkl'
    model_name = '../models/' + ml_type + '_model'+name_addition+'.pkl'
    joblib.dump(model, model_name)
    joblib.dump(result, result_name)
    return result, model, score
