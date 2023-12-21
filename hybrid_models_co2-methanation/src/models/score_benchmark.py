"""Model evaluation."""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_percentage_error)

from mechanistic_part.reaction_rate import LHHW


def compare_with_LHHW_data(df):
    """Compare noisy data with LHHW data to get the best possible model."""
    df = pd.DataFrame(df)
    X, y = df.iloc[:, 0:-1].values, df.iloc[:, -1].values
    # call LHHW
    y_LHHW = LHHW(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
    # make output

    RMSE_train = mean_squared_error(y, y_LHHW)
    MAPE_train = mean_absolute_percentage_error(y, y_LHHW)
    R2_train = r2_score(y, y_LHHW)
    score_LHHW = ({'RMSE LHHW': RMSE_train,
                   'MAPE LHHW': MAPE_train,
                  'R2 LHHW': R2_train})
    score_LHHW = pd.DataFrame(score_LHHW, index=[0])
    return score_LHHW


def get_baseline_model(df_train, df_test):
    """Create baseline model."""
    # split data
    X_train = df_train.iloc[:, 0:-1].values
    y_train = df_train.iloc[:, -1].values
    X_test = df_test.iloc[:, 0:-1].values
    y_test = df_test.iloc[:, -1].values
    dummy_regr = LinearRegression()
    dummy_regr.fit(X_train, y_train)
    y_train_dumnmy_regr = dummy_regr.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_train_dumnmy_regr)
    MAPE_train = mean_absolute_percentage_error(y_train, y_train_dumnmy_regr)
    R2_train = r2_score(y_train, y_train_dumnmy_regr)
    y_test_dumnmy_regr = dummy_regr.predict(X_test)
    MSE_test = mean_squared_error(y_test, y_test_dumnmy_regr)
    MAPE_test = mean_absolute_percentage_error(y_test, y_test_dumnmy_regr)
    R2_test = r2_score(y_test, y_test_dumnmy_regr)
    score_baseline = {'MSE Training': MSE_train,
                      'MSE Validation': MSE_test,
                      'MAPE Training': MAPE_train,
                      'MAPE Validation': MAPE_test,
                      'R2 Training': R2_train,
                      'R2 Validation': R2_test}
    return score_baseline
